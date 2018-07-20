"""
择时 notebook 所用函数。 来自07-09择时notebook
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


def getWeights_FFD(d, thres):
    """
    计算指定阶数的差分器在指定长度上的omega值。
    """
    w = [1.]
    size_max = 1000
    for k in range(1, size_max):
        w_ = -w[-1] / k * (d-k+1)
        if abs(w_) < thres:
            # 如果权重的绝对值，则就此停止
            print("reach threshold=%f, d=%f, k=%d, w_=%f" %(thres, d, k, w_))
            break
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1,1)
    return w

def fracDiff_FFD(series, d, thres = 1e-5):
    """
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    """
    #1) Compute weights for the longest series
    w = getWeights_FFD(d, thres)
    width = len(w)-1
    #2) Apply weights to values
    df ={}
    for name in series.columns:
        seriesF,df_ = series[[name]].fillna(method = 'ffill').dropna(),pd.Series()
        for iloc1 in tqdm(range(width,seriesF.shape[0])):
            loc0,loc1 = seriesF.index[iloc1-width],seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]):
                continue # exclude NAs
            df_[loc1] = np.dot(w.T,seriesF.loc[loc0:loc1])[0,0]
        df[name] = df_.copy(deep = True)
    df = pd.concat(df,axis = 1)
    return df

def getTEvents(gRaw, h):
    """
    以h为阈值，对gRaw 这个series进行采样。 CUSUM方法，p66.

    gRaw: 是Series，是close的series
    h: 这里的h是阈值
    """
    tEvents, sPos, sNeg = [], 0, 0
    diff = np.log(gRaw).diff().dropna().abs()
    # 书里是 diff = gRaw.diff()
    for i in tqdm(diff.index[1:]):
        try:
            pos, neg = float(sPos + diff.loc[i]), float(sNeg + diff.loc[i])
        except Exception as e:
            print(e)
            print(sPos + diff.loc[i], type(sPos + diff.loc[i]))
            print(sNeg + diff.loc[i], type(sNeg + diff.loc[i]))
            break
        sPos, sNeg = max(0., pos), min(0., neg)
        # 如果sNeg 突破了下界，或者sPos 突破了上届，就采样。
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)


def getDailyVol_origin(close, span0=100, num_days=1):
    """
    计算每天的volatility。计算易变性是为了动态的调整阈值，固定的阈值会导致有时候目标太高，有时候目标太低。
    【只是没有休市的情况，每个自然日都是交易日】
    """
    print("标的交易无休息日的情况。")
    # daily vol reindexed to close
    # 对于某一行来说，将index的日期向前推1天，得到符合小于该日期的最大的index，对我们来说是
    # array([     0,      0,      0, ..., 240, 241, ..., 376557, 376558, 376559], dtype=int64)
    df0 = close.index.searchsorted(close.index-pd.Timedelta(days=num_days))
    # 去掉头部=0的行，因为这些行没找到index
    df0 = df0[df0>0]
    # df0 是Series，index是日期，values是该日期在close.index里能找到的1天前的日期，如果没有，则向更早推（比如周末的时候）
    df0 = (pd.Series(close.index[df0-1], index=close.index[close.shape[0]-df0.shape[0]:]))
    try:
        # index不变，value变成num_days天的涨幅.
        df0 = close.loc[df0.index]/close.loc[df0.values].values-1 # daily rets
    except Exception as e:
        print('error: {%s}\nplease confirm no duplicate indices' %(e))
    # 此处的span不是窗口长度，而是用于计算alpha的, alpha = 2/(1+span)。
    # http://pandas.pydata.org/pandas-docs/stable/computation.html#exponentially-weighted-windows
    df0 = df0.ewm(span=span0).std().rename('dailyVol')
    return df0

def getDailyVol(close, span0=100, num_days=5):
    """
    【有休息日的情况】由于A股存在休市的情况，所以需要根据tEvents的时间对numDays进行调整】
    """
    print("标的交易有休息日的情况。")
    tmp_tEvents = close.index
    myshift = num_days * (4 * 60)
    tmp_tEvents = tmp_tEvents.tolist()
    tmp_tEvents = [None] * myshift + tmp_tEvents[myshift:]   # 使用None填充头部
    tmp_tEvents = pd.DatetimeIndex(tmp_tEvents)
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
    # 此处的span不是窗口长度，而是用于计算alpha的, alpha = 2/(1+span)。
    # http://pandas.pydata.org/pandas-docs/stable/computation.html#exponentially-weighted-windows
    df0 = df0.ewm(span=span0).std().rename('dailyVol')
    return df0

def getDailyVol_agu(close, span0=100, num_days=1, df_calendar=None):
    """
    【有休息日的情况】由于A股存在休市的情况，所以需要根据tEvents的时间对numDays进行调整】
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
    # 此处的span不是窗口长度，而是用于计算alpha的, alpha = 2/(1+span)。
    # http://pandas.pydata.org/pandas-docs/stable/computation.html#exponentially-weighted-windows
    df0 = df0.ewm(span=span0).std().rename('dailyVol')
    return df0


def applyPtSlOnT1(close, events, ptSl, molecule):
    """
    p72，实现了triple-barrier

    close: series，价格
    events: dataframe，有两列，t1：竖直方向的边界的时间，如果是np.nan，则没有竖直边界，trgt：水平边界的unit width
    ptSl: 有两个值的list，[0]是 上水平边界的width，为0则没有，[1]是下水平边界的width，为0则没有
    molecule: events哪些index将被抽取，用于并行，没被抽取的就不处理。

    输出是dataframe，包含了3个边界是何时被触碰的。
    """
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    if ptSl[0] > 0:
        # pt 是 events_ 每个时刻的上边界
        pt = ptSl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index)  # NaNs
    if ptSl[1] > 0:
        # sl 是 events_ 每个时刻的下边界
        sl = -ptSl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index)  # NaNs
    for loc, t1 in tqdm(events_['t1'].fillna(close.index[-1]).iteritems()):
        # loc 是行的index，是一个时间戳，t1是值，是t1列的值，也是个时间戳
        df0 = close[loc:t1]  # path prices，取出起始时间到竖直边界这一段的series
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']  # path returns，计算该段时间内每个时间戳对应的return
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()  # earliest stop loss，找出其中最早达到下边界的index，如果不存在则是NAT
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()  # earliest profit taking，找出其中最早达到上边界的index
    return out


"""
mpPandasObj 是书中自己定义的一个方法，在20.7
并行执行 applyPtSlOnT1, close, events, ptSl都被传进了func做参数。

输出是一个dataframe，有以下几列：
t1：第一个barrier被触碰到的时候的时间戳
trgt：被用于产生水平方向的barrier的target

pdObj: 'molecule': Name of argument used to pass the molecule, events.index: List of atoms that will be grouped into molecules
"""
def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    """
    p75，找出第一个边界被触及时的时间 同时还过滤了trgt太小的label。
    side这个参数在书里是没有的，是notebook里自己添加的。
    side：Series，index是满足特定条件的时间t，比如发生了上穿越和下穿越的时间t

    close：价格series
    tEvents: 这个是采样之后的series
    ptSl: 一个数值，水平方向的两个边界的宽度，0则表示没有水平边界
    trgt: series，这个用的是最近一段时间的volatility
    minRet：用于对trgt进行过滤
    t1: False，表示不启动竖直方向的边界，还可以设为series，里面是竖直方向的时间戳

    """
    # 1) get target
    print("getEvents() start.")
    print("1st. get target.")
    trgt = trgt.loc[tEvents]
    print("1st. 只保留大于minRet的target.")
    trgt = trgt[trgt > minRet]  # minRet
    # 2) get t1 (max holding period)
    print("2nd. get t1 (max holding period).")
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
    # 3) form events object, apply stop loss on t1
    print("3rd. form events object, apply stop loss on t1.")
    if side is None:
        side_, ptSl_ = pd.Series(1., index=trgt.index), [ptSl[0], ptSl[0]]
    else:
        side_, ptSl_ = side.loc[trgt.index], ptSl[:2]
    print("3rd. 创建events. ")
    events = (pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt']))
    print("3rd. 使用mpPandasObj() 创建df0.")
    if True:
        print("不使用并行，不使用mpPandasObj().")
        # print("close:")
        # cprint(close)
        # print("events:")
        # cprint(events)
        # print('ptSl_:')
        # cprint(ptSl_)
        # print("molecule:")
        # cprint(events.index)
        df0 = applyPtSlOnT1(close, events, ptSl_, events.index)
    else:
        df0 = mpPandasObj(func=applyPtSlOnT1, pdObj=('molecule', events.index),
                          numThreads=numThreads, close=close, events=events, ptSl=ptSl_)
    print("3rd. df0创建完毕. ")
    events['t1'] = df0.dropna(how='all').min(axis=1)  # pd.min ignores nan
    if side is None:
        events = events.drop('side', axis=1)
    print("getEvents() 执行完毕. ")
    return events

def addVerticalBarrier_origin(tEvents, close, numDays=5):
    """
    返回一个series：索引是tEvents的索引，值是该索引（时间戳）所对应的下一个bar的时间戳或者是经过了指定天numDays后的时间戳。
    【全年交易，没有休息日的情况。】
    """
    print("标的交易无休息日的情况。")
    t1 = close.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
    t1 = t1[t1<close.shape[0]]
    t1 = (pd.Series(close.index[t1], index=tEvents[:t1.shape[0]]))
    return t1

def addVerticalBarrier(tEvents, close, numDays=5):
    """
    返回一个series：索引是tEvents的索引，值是该索引（时间戳）所对应的下一个bar的时间戳或者是经过了指定天numDays后的时间戳。

    【有休息日的情况】由于A股存在休市的情况，所以需要根据tEvents的时间对numDays进行调整。
    """
    print("标的交易有休息日的情况。")
    tmp_tEvents = tEvents.copy()
    myshift = numDays * (4 * 60)
    tmp_tEvents = tmp_tEvents.tolist()
    tmp_tEvents = tmp_tEvents[myshift:] + [x+pd.Timedelta(days=numDays) for x in tmp_tEvents[-myshift:]]
    # 不能使用None填充末尾，因为对None使用searchsorted会找到index=0，改为用...
    tmp_tEvents = pd.DatetimeIndex(tmp_tEvents)
    t1 = close.index.searchsorted(tmp_tEvents)
    t1 = t1[t1 < close.shape[0]]   # 如果没符合条件的，searchsorted会返回长度，所以此处删去values为长度的行
    t1 = (pd.Series(close.index[t1], index=tEvents[:t1.shape[0]]))
    return t1

def addVerticalBarrier_agu(tEvents, close, numDays=5, df_calendar=None):
    """
    从 s_tt 中获取指定日期numDays的交易日间隔对应多少个自然日的间隔，然后在close里找到符合该条件的时刻。
    :param tEvents: 
    :param close: 
    :param numDays: 
    :param s_tt: Series
    :return: 一个Series，它的index是tEvents的index，value是该index时间所对应的竖直边界
    """
    s_tt = df_calendar["n"+str(numDays)]
    print("标的交易存在休息日的情况。")
    tmp_tEvents = list()
    for datetime in tqdm(tEvents.tolist()):
        date = str(datetime.date())
        shift_day = s_tt[date]
        new_datetime = datetime + pd.Timedelta(days=shift_day)
        tmp_tEvents.append(new_datetime)
    tmp_tEvents = pd.DatetimeIndex(tmp_tEvents)
    t1 = close.index.searchsorted(tmp_tEvents)
    t1 = t1[t1 < close.shape[0]]  # 如果没符合条件的，searchsorted会返回长度，所以此处删去values为长度的行
    t1 = (pd.Series(close.index[t1], index=tEvents[:t1.shape[0]]))
    return t1


def getBins(events, close):
    '''
    给样本打标签。
    【此处与书中的不一样】
    分只计算size，和同时计算size和side两种情况

    触碰到triple barrier时的价格相对于初始价格的涨幅，大于0则为1，小于0为-1，等于0为0.
    events：从getEvents() 获取结果，dataframe有3列，t1，sl（触碰下边界的时间，可能是NAT），pt（触碰上边界的时间）。
    
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    '''
    # 1) prices aligned with events
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    # 使用新值来重建原始Series的索引，bfill: use next valid observation to fill gap
    px = close.reindex(px, method='bfill')
    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    if 'side' in events_:
        out['ret'] *= events_['side']           # meta-labeling
    out['bin'] = np.sign(out['ret'])
    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0     # meta-labeling
    return out

def dropLabels(events, minPct=.05):
    """
    p81，如果存在3类及以上，并且有类别的占比不足 5%，则把它们删除，之后重新计算占比，继续删除占比不足的类别。
    """
    # apply weights, drop labels with insufficient examples
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min()>minPct or df0.shape[0]<3:
            break
        print('dropped label: ', df0.argmin(),df0.min())
        events = events[events['bin']!=df0.argmin()]
    return events

def linParts(numAtoms, numThreads):
    """
    进行线性的分割，如果输入 linParts(10000, 3)，则返回array([    0.,  5000., 10000.])
    """
    # partition of atoms with a single loop
    parts = np.linspace(0, numAtoms, min(numThreads,numAtoms)+1)
    parts = np.ceil(parts).astype(int)
    return parts


def mpPandasObj(func, pdObj, numThreads=24, mpBatches=1, linMols=True, **kargs):
    '''
    Parallelize jobs, return a dataframe or series
    + func: function to be parallelized. Returns a DataFrame
    + pdObj[0]: Name of argument used to pass the molecule
    + pdObj[1]: List of atoms that will be grouped into molecules
    + kwds: any other argument needed by func

    Example: df1=mpPandasObj(func,('molecule',df0.index),24,**kwds)
    '''
    # if linMols:parts=linParts(len(argList[1]),numThreads*mpBatches)
    # else:parts=nestedParts(len(argList[1]),numThreads*mpBatches)
    if linMols:
        # 进行线性的分割，如果为False，则进行嵌套的分割
        parts = linParts(len(pdObj[1]), numThreads * mpBatches)
    else:
        parts = nestedParts(len(pdObj[1]), numThreads * mpBatches)
    # parts 的取值例子如下: array([    0.,  5000., 10000.])
    jobs = []
    print("创建任务的list。")
    for i in range(1, len(parts)):
        job = {pdObj[0]: pdObj[1][parts[i - 1]:parts[i]], 'func': func}
        job.update(kargs)
        jobs.append(job)
    print("开始并行处理任务list。")
    cprint(jobs)
    if numThreads == 1:
        print("线程数量=1。")
        # 如果使用的线程数量=1，按顺序一个一个地执行jobs里的任务
        out = processJobs_(jobs)
    else:
        print("线程数量大于1。")
        # 并行执行 【todo】会卡住，原因还没找到
        out = processJobs(jobs, numThreads=numThreads)
    cprint(out)
    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series()
    else:
        return out
    print("将每个任务的结果拼接在一起。")
    for i in out:
        df0 = df0.append(i)
    print("拼接完成，将最终结果按索引排序。")
    df0 = df0.sort_index()
    return df0

def processJobs_(jobs):
    # Run jobs sequentially, for debugging
    out=[]
    count = 1
    for job in jobs:
        print("job: %d" %(count))
        count += 1
        out_=expandCall(job)
        out.append(out_)
    return out


def reportProgress(jobNum, numJobs, time0, task):
    # Report progress as asynch jobs are completed
    msg = [float(jobNum)/numJobs, (time.time()-time0)/60.]
    msg.append(msg[1]*(1/msg[0]-1))
    timeStamp = str(dt.datetime.fromtimestamp(time.time()))
    msg = timeStamp+' '+str(round(msg[0]*100,2))+'% '+task+' done after '+ \
        str(round(msg[1],2))+' minutes. Remaining '+str(round(msg[2],2))+' minutes.'
    if jobNum<numJobs:
        sys.stderr.write(msg+'\r')
    else:
        sys.stderr.write(msg+'\n')
    return

def processJobs(jobs,task=None,numThreads=24):
    # Run in parallel.
    # jobs must contain a 'func' callback, for expandCall
    if task is None:
        task=jobs[0]['func'].__name__
    pool = mp.Pool(processes=numThreads)
    outputs, out,time0 = pool.imap_unordered(expandCall,jobs), [],time.time()
    # Process asyn output, report progress
    for i,out_ in enumerate(outputs,1):
        out.append(out_)
        reportProgress(i,len(jobs),time0,task)
    pool.close()
    pool.join() # this is needed to prevent memory leaks
    return out

def expandCall(kargs):
    # Expand the arguments of a callback function, kargs['func']
    func=kargs['func']
    del kargs['func']
    out=func(**kargs)
    return out


"""以上跟打标签有关，以下与特征构建和建模有关。"""
def get_up_cross(df, col):
    """
    筛选出同时满足以下条件的行（也就是发生了上边界穿越的地方）：
    1.该行的上一行的col 值小于该行的upper值
    2.该行的col值大于该行的upper值
    """
    # col is price column
    crit1 = df[col].shift(1) < df.upper    # shift(1), 移动之后，t+1行的值是原来的第t行的值
    crit2 = df[col] > df.upper
    return df[col][(crit1) & (crit2)]

def get_down_cross(df, col):
    """
    筛选出同时满足以下条件的行：
    1.该行的上一行的col值大于该行的lower值
    2.该行的col值小于该行的lower值
    """
    # col is price column
    crit1 = df[col].shift(1) > df.lower
    crit2 = df[col] < df.lower
    return df[col][(crit1) & (crit2)]

def bbands(price, window=None, width=None, numsd=None):
    """ 
    p68 和 p82的习题中给出了Bollinger bands， 1.05，0.95倍的ave作为band的上下界。
    如果使用numsd，则使用ave加减一定倍数的sd来作为上下界限。
    只使用发生了越界时的样本来进行建模和预测。
    
    price 此处是close
    ave 是一段时间（长度为window）的price均值
    upband 是 (1+width) * ave
    dnband 是 (1-width) * ave
    returns average, upper band, and lower band
    """
    ave = price.rolling(window).mean()
    sd = price.rolling(window).std(ddof=0)
    # ddof: Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements.
    if width:
        upband = ave * (1+width)
        dnband = ave * (1-width)
        return price, np.round(ave,3), np.round(upband,3), np.round(dnband,3)
    if numsd:
        upband = ave + (sd*numsd)
        dnband = ave - (sd*numsd)
        return price, np.round(ave,3), np.round(upband,3), np.round(dnband,3)


def macd(close, short_window=500, long_window=1000, signalperiod=350):
    # 短周期12天，大约500跨度，长周期26天，大约1000跨度，dea9天，大约350跨度
    # hist = macd - signal，分别指什么，macd是DIF，signal是DEA。
    macd, signal, hist = talib.MACD(close.values, fastperiod=short_window, slowperiod=long_window, signalperiod=signalperiod)
    tmp_df = (pd.DataFrame(index=close.index).assign(macd=macd)
      .assign(signal=signal).assign(hist=hist).assign(close=close.values))
    return tmp_df

def macd_signals(macd_df):
    # macd_signal是信号列，buy=1，sell=-1，没信号=0
    df0 = macd_df.copy()
    df0["DIF_diff"] = df0.macd.diff()  # DIF_t - DIF_(t-1)
    df0.dropna(inplace=True)
    df0.loc[:, "macd_signal"] = 0    # 初始化全部为0，下面分配1，-1
    buyct1 = df0['hist'].shift(1) < 0
    buyct2 = df0['hist'] > 0
    buyct3 = df0["macd"] > 0
    buyct4 = df0["DIF_diff"] > 0
    sellct1 = df0['hist'].shift(1) > 0
    sellct2 = df0['hist'] < 0
    sellct3 = df0["macd"] < 0
    sellct4 = df0["DIF_diff"] < 0
    df0['macd_signal'][buyct1&buyct2&buyct3&buyct4] = 1
    df0['macd_signal'][sellct1&sellct2&sellct3&sellct4] = -1
    return df0


def returns(s):
    arr = np.diff(np.log(s))
    return (pd.Series(arr, index=s.index[1:]))

def df_rolling_autocorr(df, window, lag=1):
    """
    Compute rolling column-wise autocorrelation for a DataFrame.
    """
    return (df.rolling(window=window).corr(df.shift(lag))) # could .dropna() here


"""

"""


"""以下和uniqueness， sequential bootstrap有关"""
def mpNumCoEvents(closeIdx, t1, molecule):
    """
    p87, 4.1 Estimate the uniqueness of a label.
    计算了the number of labels concurrent at t，但是算的有错。
    
    Compute the number of concurrent events per bar.
    +molecule[0] is the date of the first event on which the weight will be computed
    +molecule[-1] is the date of the last event on which the weight will be computed
    AVERAGE UNIQUENESS OF A LABEL 61
    Any event that starts before t1[molecule].max() impacts the count.
    """
    #1) find events that span the period [molecule[0],molecule[-1]]
    t1 = t1.fillna(closeIdx[-1]) # unclosed events still must impact other weights
    t1 = t1[t1 >= molecule[0]] # events that end at or after molecule[0]
    t1 = t1.loc[:t1[molecule].max()] # events that start at or before t1[molecule].max()
    #2) count events spanning a bar
    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=closeIdx[iloc[0]:iloc[1]+1])
    for tIn,tOut in t1.iteritems():
        count.loc[tIn:tOut] += 1.
    return count.loc[molecule[0]:t1[molecule].max()]

def mpSampleTW(t1, numCoEvents, molecule):
    """ 
    p89, 4.2 Estimate the average uniqueness of a label.
    计算每个label的uniqueness，方法是计算该label所跨越的所有时刻t的uniqueness，然后取均值。
    
    Derive average uniqueness over the event's lifespan
    """
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn] = (1./numCoEvents.loc[tIn:tOut]).mean()
    return wght

def getIndMatrix(barIx, t1):
    """ 
    p91, 4.3 Build an indicator matrix.
    使用index of the bars 和 Series t1 创建一个indicator matrix 指标矩阵。
    t1 就是上面已经用到的，index是该样本的时间截面，value是该样本的观察窗口结束的时间截面。
    输出是一个只含有0和1的矩阵，indicate what bars influence the label for each observation.
    
    Get indicator matrix
    """
    indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[0]))
    # t0是index，t1是value
    for i, (t0, t1) in tqdm(enumerate(t1.iteritems())):
        indM.loc[t0:t1, i] = 1.
    return indM

def getAvgUniqueness(indM):
    """ 
    p92, 4.4 Compute average uniqueness.
    计算4.3得出的指标矩阵 中每个样本的average uniqueness。
    
    Average uniqueness from indicator matrix
    """
    c = indM.sum(axis=1) # concurrency
    u = indM.div(c, axis=0) # uniqueness
    avgU = u[u>0].mean() # average uniqueness
    return avgU

def seqBootstrap(indM, sLength=None):
    """ 
    p92, 4.5 Return sample from sequential bootstrap
    输出是被采样的样本的index，结果中可能存在重复的样本。
    
    Generate a sample via sequential bootstrap
    """
    if sLength is None:
        sLength = indM.shape[1]
    phi = []
    while len(phi) < sLength:
        avgU = pd.Series()
        for i in indM:
            indM_ = indM[phi+[i]] # reduce indM
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
        prob = avgU/avgU.sum() # draw prob
        print("phi: %s, avgU: %s, prob: %s" %(str(phi), str(avgU), str(prob)))
        phi += [np.random.choice(indM.columns, p=prob)]
    return phi

# 以下是一个例子
def main():
    # 4.6
    t1 = pd.Series([2,3,5],index = [0,2,4]) # t0,t1 for each feature obs
    barIx = range(t1.max()+1) # index of bars
    indM = getIndMatrix(barIx,t1)
    phi = np.random.choice(indM.columns,size = indM.shape[1])
    print(phi)
    print('Standard uniqueness:',getAvgUniqueness(indM[phi]).mean())
    phi = seqBootstrap(indM)
    print(phi)
    print('Sequential uniqueness:',getAvgUniqueness(indM[phi]).mean())
    return

def getRndT1(numObs,numBars,maxH):
    # 4.7
    #  random t1 Series
    t1 = pd.Series()
    for i in xrange(numObs):
        ix = np.random.randint(0,numBars)
        val = ix+np.random.randint(1,maxH)
        t1.loc[ix] = val
    return t1.sort_index()

def auxMC(numObs,numBars,maxH):
    # 4.8
    # Parallelized auxiliary function
    t1 = getRndT1(numObs,numBars,maxH)
    barIx = range(t1.max()+1)
    indM = getIndMatrix(barIx,t1)
    phi = np.random.choice(indM.columns, size=indM.shape[1])
    stdU = getAvgUniqueness(indM[phi]).mean()
    phi = seqBootstrap(indM)
    seqU = getAvgUniqueness(indM[phi]).mean()
    return { 'stdU':stdU, 'seqU':seqU }

def mainMC(numObs = 10, numBars = 100, maxH = 5, numIters = 1E6, numThreads = 24):
    # Monte Carlo experiments
    jobs = []
    for i in xrange(int(numIters)):
        job ={'func':auxMC, 'numObs':numObs, 'numBars':numBars, 'maxH':maxH}
        jobs.append(job)
    if numThreads == 1:
        out = processJobs_(jobs)
    else:
        out = processJobs(jobs,numThreads = numThreads)
    print(pd.DataFrame(out).describe())
    return

def mpSampleW(t1, numCoEvents, close, molecule):
    """ 
    4.10
    Series.iteritems(): Lazily iterate over (index, value) tuples
    根据样本在观察窗口上的return和并发样本个数，计算它的权重.
    
    Derive sample weight by return attribution
    """
    ret = np.log(close).diff() # log-returns, so that they are additive
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].iteritems():
        # 这个是书p96的表达式sum() 里的每一项，都是某个时刻t的return（一个时间长度），除以该时刻覆盖的label的个数，求和之后取绝对值，
        wght.loc[tIn] = (ret.loc[tIn:tOut]/numCoEvents.loc[tIn:tOut]).sum()
    return wght.abs()

def getTimeDecay(tW, clfLastW=1.):
    """ 
    4.11
    
    
    apply piecewise-linear decay to observed uniqueness (tW)
    newest observation gets weight = 1, oldest observation gets weight = clfLastW
    """
    clfW = tW.sort_index().cumsum()
    if clfLastW >= 0:
        slope = (1. - clfLastW)/clfW.iloc[-1]
    else:
        slope = 1./( (clfLastW + 1) * clfW.iloc[-1] )
    const = 1. - slope * clfW.iloc[-1]
    clfW = const + slope * clfW
    clfW[clfW<0] = 0
    print(const, slope)
    return clfW


class SIGNAL_FAST_SLOW:
    @staticmethod
    def get_up_cross(df):
        crit1 = df.fast.shift(1) < df.slow
        crit2 = df.fast > df.slow
        return df.fast[(crit1) & (crit2)]
    @staticmethod
    def get_down_cross(df):
        crit1 = df.fast.shift(1) > df.slow
        crit2 = df.fast < df.slow
        return df.fast[(crit1) & (crit2)]











