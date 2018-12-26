import pandas as pd
import numpy as np
import time

#基于论文order imbalance 论文实现高频交易策略 
#该程序是用于计算论文中回归函数的所有指标 ORI VOI  Rt Std等 
# 每个函数计算一种指标 最后用于回归的指标只有四个 
# 有的指标是复合指标 通过调用其他的简单指标计算得到

def func_delta_vol_bid_price(df):
    # time_start=time.time();#time.time()为1970.1.1到当前时间的毫秒数  
    price = df['bp1']
    volume = df['bv1']
    volume_delta_df =pd.DataFrame(volume.values[1:] - volume.values[0:-1],index = price.index[1:],columns = ['volume_delta'])
    price_delta_df = pd.DataFrame(price.values[1:] - price.values[0:-1],index = price.index[1:],columns = ['ask_price_delta'])
	 #价格相等
    index_eqal = list(price_delta_df[(price_delta_df == 0).values].index)
	 #大于上一时刻的价格
    index_bigger = list(price_delta_df[(price_delta_df > 0).values].index)
	 #初始化
    delta_vol_price = pd.DataFrame(np.zeros((len(price_delta_df),1)),index = price.index[1:],columns = ['delta_vol_price'])
    volume = volume[1:]
    # 通过将series传入dataframe这种方式操作，速度更快，但会保留一些空值，用fillna填上即可（从逻辑上是跟paper的方法一致的）
    temp = pd.concat([volume[index_bigger], volume_delta_df.loc[index_eqal,'volume_delta']])
    delta_vol_price['delta_vol_price'] = temp
    delta_vol_price = delta_vol_price.fillna(0)
    # 关闭计时程序
    # time_end=time.time();#time.time()为1970.1.1到当前时间的毫秒数  
    # print ('新方法')
    # print(time_end-time_start)
    return delta_vol_price

def func_delta_vol_ask_price(df):
    # time_start=time.time();#time.time()为1970.1.1到当前时间的毫秒数 
    price = df['ap1']
    volume = df['av1']
    volume_delta_df =pd.DataFrame(volume.values[1:] - volume.values[0:-1],index = price.index[1:],columns = ['volume_delta'])
    price_delta_df = pd.DataFrame(price[1:].values - price[0:-1].values, index = price.index[1:],columns = ['ask_price_delta'])
    #价格相等
    index_eqal = list(price_delta_df[(price_delta_df == 0).values].index)
    #小于上一时刻的价格 
    index_smaller = list(price_delta_df[(price_delta_df < 0).values].index)
    delta_vol_price = pd.DataFrame(np.zeros((len(price_delta_df),1)),index = price.index[1:],columns = ['delta_vol_price'])
    volume = volume[1:]
    temp = pd.concat([volume[index_smaller], volume_delta_df.loc[index_eqal,'volume_delta']])
    delta_vol_price.loc[index_smaller,'delta_vol_price'] = temp
    delta_vol_price = delta_vol_price.fillna(0)
    # 关闭计时程序
    # time_end=time.time();#time.time()为1970.1.1到当前时间的毫秒数  
    # print ('旧方法')
    # print(time_end-time_start)
    return delta_vol_price

def func_VOI(df):
	'''
	计算 volume order imbalance 
	'''

	delta_bid_vol_price = func_delta_vol_bid_price(df)
	
	delta_ask_vol_price = func_delta_vol_ask_price(df)
	VOI = delta_bid_vol_price - delta_ask_vol_price 
	VOI.columns = ['VOI']
	return VOI




# 下面这几个关于func_mean_mid_price的函数有可能都写错了, 后面要检查一下
def func_mean_mid_price1(df,K):
    # 计算未来K个tick的中间价均值和此刻tick的中间价的差值
    bid_price = df['bp1']
    ask_price = df['ap1']
    mid_price = (bid_price.values + ask_price.values)/2 #中间价
    # 转换成DataFrame，方便使用rolling函数，滚动计算均值    
    mid_price_df = pd.DataFrame(mid_price,index = range(len(mid_price)), columns = ['mid_price'])
    mean_mid_price =mid_price_df.rolling(window=K, min_periods= K).mean()
    delta_mid_price = mean_mid_price.iloc[K-1:,0].values - mid_price[:len(mid_price)-K+1]
    # 得到mid_price和未来K个tick均值的差，加上时间索引，并返回
    delta_mid_price_df = pd.DataFrame(delta_mid_price, index = df.index.values[:-K+1], columns = ['delta_M'])
    return delta_mid_price_df


def func_mean_mid_price_Jcd_revised(df,K):
    # 计算未来K个tick的中间价均值和此刻tick的中间价的差值
    bid_price = df['bp1']
    ask_price = df['ap1']
    mid_price = (bid_price.values + ask_price.values)/2 #中间价
    # 转换成DataFrame，方便使用rolling函数，滚动计算均值    
    mid_price_df = pd.DataFrame(mid_price,index = range(len(mid_price)), columns = ['mid_price'])
    mean_mid_price =pd.rolling_mean(mid_price_df,window=K, min_periods= K)
    delta_mid_price = mean_mid_price.iloc[K-1:,0].values - mid_price[:len(mid_price)-K+1]
    # 得到mid_price和未来K个tick均值的差，加上时间索引，并返回
    delta_mid_price_df = pd.DataFrame(delta_mid_price, index = df.index.values[:-K+1], columns = ['delta_M'])
    return delta_mid_price_df


def func_mean_mid_price(df, forecast_window, tick_size):
    '''
	Average mid-price change :平均中间报价变动： 我暂时没有检查这个函数和前面几个类似函数之间的区别是什么
	'''
    
    K = forecast_window * tick_size
    bid_price = df['bp1']
    ask_price = df['ap1']
    #得到一列中间价
    mid_price = (bid_price.values + ask_price.values)/2 
    
    # 计算未来forecast_window - 1个相应tick的价格总和
#     先生成一列sum_mid_price 长度为 mid_price - K; 以0值来占位
    sum_mid_price = np.zeros((len(mid_price)- K))
    
    for i in range(1,forecast_window):
        #后续检查一下这句话是不是对的，应该用哪一个
#         i = i * num_of_tick2synthesis
#         i = i 
        sum_mid_price = sum_mid_price + mid_price[i:i-K]
    # 补上第forecast_window相应tick的数据
    sum_mid_price = sum_mid_price + mid_price[K:]
    # 求均值
    sum_mid_price = sum_mid_price/forecast_window
    # 未来20日的均价减去中间价
    mean_mid_price = sum_mid_price - mid_price[0:-K] 
    #DataFrame格式输出
    mean_mid_price_df = pd.DataFrame(np.array(mean_mid_price).reshape((len(mid_price)-K,1)),index = bid_price.index[0:-K],columns = ['mean_mid_price'])
    return mean_mid_price_df









def func_ORI(df):
	'''
	order imbalance ratio 

	'''
	bid_vol = df['bv1']
	ask_vol = df['av1']
	rho = (bid_vol.values  - ask_vol.values)/(bid_vol.values + ask_vol.values)
	rho_df = pd.DataFrame(rho,index = bid_vol.index,columns = ['ORI'])
	return rho_df

def func_mid_price(df):
	'''
	计算中间价: 0.5 * (bid + ask)
	'''
	bid_price = df['bp1']
	ask_price = df['ap1']
	mid_price = (bid_price.values + ask_price.values)/2 #中间价
	mid_price_df = pd.DataFrame(mid_price,index = bid_price.index,columns = ['mid_price'])
	return mid_price_df




def func_mean_TP(df):
	'''
    TP t
	average trade price 
	'''

	amount = df['amount']
	volume = df['volume']
	mid_price = (df['bp1'].values[0] + df['ap1'].values[0])/2
	mean_TP = []
	mean_TP.append(mid_price)
	for i in range(1,len(volume)):
		if volume[i] == volume[i-1]:
			mean_TP.append(mean_TP[-1])
		else :
			item = (amount[i] - amount[i-1])/(volume[i] - volume[i-1])/300
			mean_TP.append(item)
	mean_TP_df = pd.DataFrame(mean_TP,index = volume.index,columns = ['mean_TP'])
	return mean_TP_df

def func_mid_price_basis(df):   
	'''
    Rt
	mid price basis = average trade price - average mid price 
	'''
	mid_price_df = func_mid_price(df)
	mean_TP_df = func_mean_TP(df)
	mean_mid_price = []
	mean_mid_price.append(list(mid_price_df.values[0]))
	
	mean_mid_price = mean_mid_price + list((mid_price_df.values[0:-1] + mid_price_df.values[1:])/2)
	
	mid_price_basis = mean_TP_df.values - mean_mid_price

	mid_price_basis_df = pd.DataFrame(mid_price_basis,index = mean_TP_df.index, columns =['Rt'])
	return mid_price_basis_df[1:]

def func_bid_ask_spread(df):
	'''
	bid-ask spread 
	'''
	bid_price = df['bp1']
	ask_price = df['ap1']
	bid_ask_spread = ask_price - bid_price 
	bid_ask_spread_df = pd.DataFrame(bid_ask_spread,index = df.index,columns = ['St'])
	return bid_ask_spread_df


def func_delta_vol_bid_price_5(df):
#计算5档行情的delta of volume at bid price 
    for i in range(5):
        label_price = 'bp'+str(i+1)
        label_volume = 'bv'+str(i+1)
        price = df[label_price]
        volume = df[label_volume]

    price_delta = price[1:].values - price[0:-1].values
    volume_delta_df =pd.DataFrame(volume.values[1:] - volume.values[0:-1],index = price.index[1:],columns = ['volume_delta'])
    price_delta_df = pd.DataFrame(np.array(price_delta),index = price.index[1:],columns = ['bid_price_delta'])
	
	#价格相等
    index_eqal = price_delta_df[(price_delta_df == 0).values].index
	#大于上一时刻的价格
    index_bigger = list(price_delta_df[(price_delta_df > 0).values].index)
	#初始化
    delta_vol_price = pd.DataFrame(np.zeros((len(price_delta_df),1)),index = price.index[1:],columns = ['delta_vol_price'])
    volume = volume[1:]
    delta_vol_price.loc[index_bigger,'delta_vol_price'] = volume[index_bigger]
    delta_vol_price.loc[index_eqal,'delta_vol_price'] = volume_delta_df.loc[index_eqal,'volume_delta']
    return delta_vol_price

def func_delta_vol_ask_price_5(df):
	'''
	计算5档行情的delta of volume at ask price 

	'''
	
	price = df['ap1']
	volume = df['av1']

	price_delta = price[1:].values - price[0:-1].values
	volume_delta_df =pd.DataFrame(volume.values[1:] - volume.values[0:-1],index = price.index[1:],columns = ['volume_delta'])
	price_delta_df = pd.DataFrame(np.array(price_delta),index = price.index[1:],columns = ['ask_price_delta'])
	#价格相等
	index_eqal = price_delta_df[(price_delta_df == 0).values].index
	#小于上一时刻的价格 
	index_smaller = list(price_delta_df[(price_delta_df < 0).values].index)

	delta_vol_price = pd.DataFrame(np.zeros((len(price_delta_df),1)),index = price.index[1:],columns = ['delta_vol_price'])
	volume = volume[1:]
	delta_vol_price.loc[index_smaller,'delta_vol_price'] = volume[index_smaller]
	
	delta_vol_price.loc[index_eqal,'delta_vol_price'] = volume_delta_df.loc[index_eqal,'volume_delta']

	return delta_vol_price

def func_VOI_5(df):
	'''
	计算 5挡行情的volume order imbalance 
	'''

	delta_bid_vol_price = func_delta_vol_bid_price_5(df)
	
	delta_ask_vol_price = func_delta_vol_ask_price_5(df)
	VOI = delta_bid_vol_price - delta_ask_vol_price 
	VOI.columns = ['VOI']
	return VOI
