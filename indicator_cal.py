"""
择时所用的因子.

所有函数的输入都是dataframe，列的名字是open, high, low, close, money, amount，index是时间戳。
输出是Series，index是时间戳。

以下的“天”没有特别说明，均指交易日。
一天大约是100个跨度，一个小时大约24个跨度。
"""
import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import tushare as ts
import multiprocessing as mp
import datetime as dt
from cprint import cprint
from tqdm import tqdm
import talib


def cal_vwap(df):
    # vwap, 非只close
    pass

def cal_snr():
    # SNR，非只close
    pass

def cal_wdiff(df, n):
    # WDIFF，文档上没有指明是窗口是多长时间
    #todo 长度待定
    pass

def cal_mtm(df, n=1400, m=700, if_ma=False):
    # MTM, MTM_MA，mtm 是14天，均值是7天
    df_tmp = df.copy()
    if not if_ma:
        # 计算 MTM
        mtm = df_tmp.close.diff(periods=n)
        return mtm
    else:
        # 计算 MTM的均值
        mtm = df_tmp.close.diff(periods=n)
        mtm_ma = mtm.rolling(window=m).mean()
        return mtm_ma

def cal_volatility(df, n, N):
    #todo
    pass

def getDailyVol_agu(close, span0=100, num_days=1, df_calendar=None):
    """
    由于A股存在休市的情况，所以需要根据tEvents的时间对numDays进行调整，从df_calendar获取交易日历
    """
    s_tt = df_calendar["prev_n"+str(num_days)]
    print("标的交易有休息日的情况。")
    tmp_tEvents = list()
    for datetime in tqdm(close.index.tolist()):
        date = str(datetime.date())
        shift_day = s_tt[date]
        new_datetime = datetime - pd.Timedelta(days=shift_day)
        tmp_tEvents.append(new_datetime)
    tmp_tEvents = close.index
    df0 = close.index.searchsorted(tmp_tEvents)
    # 去掉头部=0的行，因为这些行没找到index
    df0 = df0[df0>0]
    # df0 是Series，index是日期，values是该日期在close.index里能找到的1天前的日期，如果没有，则向更早推（比如周末的时候）
    df0 = (pd.Series(close.index[df0-1], index=close.index[close.shape[0]-df0.shape[0]:]))
    try:
        # index不变，value变成num_days天的涨幅.
        df0 = close.loc[df0.index]/close.loc[df0.values].values-1 # daily rets
    except Exception as e:
        print('error: {%s}\nplease confirm no duplicate indices' %(e))
    # 此处的span不是窗口长度，而是用于计算指数加权窗口的alpha, alpha = 2/(1+span)。
    # http://pandas.pydata.org/pandas-docs/stable/computation.html#exponentially-weighted-windows
    df0 = df0.ewm(span=span0).std().rename('dailyVol')
    return df0

def cal_srl_corr(df, n=500):
    # 计算5天的自相关
    def returns(series):
        arr = np.log(series).diff()
        return pd.Series(arr, index=series.index)
    def rolling_corr(series, window, lag=1):
        # 计算window窗口长度上 t 与 t-1的自相关性
        return (series.rolling(window=window).corr(series.shift(lag)))
    srl_corr = rolling_corr(returns(df.close), window=n, lag=1)
    return srl_corr

def cal_roc(df, n=1200, m=600, if_ma=False):
    # ROC，n是12天，m是6天
    df_tmp = df.copy()
    if not if_ma:
        # 计算ROC
        roc = df_tmp.close.diff(periods=n)
        roc = 100 * roc.divide(df_tmp.close.shift(n))
        return roc
    else:
        # 计算ROC MA
        roc = df_tmp.close.diff(periods=n)
        roc = 100 * roc.divide(df_tmp.close.shift(n))
        roc_ma = roc.rolling(window=m).mean()
        return roc_ma

def cal_obv():
    # OBV，非只close
    pass

def cal_ma(df, n=500):
    # MA，收盘价均线，常用的有5天，10天，20天，60天。以及实验效果好的600和5000.
    df_tmp = df.copy()
    close_ma = df_tmp.close.rolling(window=n).mean()
    return close_ma

def cal_emv():
    # EMV，非只close
    pass

def cal_boll(df, n=2400, numsd_up=4.3, numsd_down=4.3):
    # 布林带，由实验得出最优窗口2400，倍数4.3，与公式中的20天，2倍不同
    df_tmp = df.copy()
    boll = df_tmp.close.rolling(window=n).mean()
    std = df_tmp.close.rolling(window=n).std()
    boll_up = boll.add(numsd_up * std)
    boll_down = boll.subtract(numsd_down * std)
    return boll, boll_up, boll_down

def cal_psy(df, n=1200, m=600, if_ma=False):
    # PSY, PSY_MA，文档上建议n=12天，m=6天
    # 使用pd.rolling_apply() 会导致windows python自动关闭，改为使用Series.rolling().apply(cust_func)
    diff_series = df.close.diff(periods=1)
    count_pos = lambda ndarray: np.where(ndarray>1)[0].size    # 只计算并返回第一列的滑动窗口正数计数值
    count_series = diff_series.rolling(window=n).apply(count_pos)
    psy = 100 * count_series / n
    if not if_ma:
        return psy
    else:
        psy_ma = psy.rolling(window=m).mean()
        return psy_ma

def cal_macd(df, n_short=700, n_long=4000, m=900):
    # MACD_DIF, MACD_DEA, MACD，实验的最优结果是700，4000，900，文档建议12天，26天，9天
    macd_dif, macd_dea, macd = talib.MACD(df.close.values, fastperiod=n_short, slowperiod=n_long,
                                    signalperiod=m)
    macd_dif = pd.Series(index=df.index, data=macd_dif)
    macd_dea = pd.Series(index=df.index, data=macd_dea)
    macd = pd.Series(index=df.index, data=macd)
    return macd_dif, macd_dea, macd

def cal_dma(df, n1=1000, n2=5000, m=1000, if_ma=False):
    # DMA, DMA_MA，文档中建议n1=10天，n2=50天，m=10天
    dma = (df.close.rolling(window=n1).mean()).subtract(df.close.rolling(window=n2).mean())
    if not if_ma:
        return dma
    else:
        dma_ma = dma.rolling(window=m).mean()
        return dma_ma

def cal_dema(df, n=1000):
    # DEMA, 文档中建议n=10天
    deam = talib.DEMA(df.close.values, timeperiod=n)
    deam = pd.Series(index=df.index, data=deam)
    return deam

def cal_alma(df, n, sigma, offset):
    # Arnaud Legoux Moving Average，默认window=9天，sigma=6，offset=0.85
    #todo 公式不明
    price = df.close
    m = offset * (n - 100)
    s = window/sigma
    pass

def cal_rsi(df, n=1400):
    # RSI, talib默认n=14天，与文档中n=6天不一样
    rsi = talib.RSI(df.close.values, timeperiod=n)
    rsi = pd.Series(index=df.index, data=rsi)
    return rsi

def cal_trix(df, n=1200):
    # TRIX，talib默认n=30天，与文档中n=12天不一样
    trix = talib.TRIX(df.close.values, timeperiod=n)
    trix = pd.Series(index=df.index, data=trix)
    return trix














