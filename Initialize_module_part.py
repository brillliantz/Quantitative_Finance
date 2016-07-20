import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib.finance as mf
from matplotlib.widgets import MultiCursor
import statsmodels.tsa.stattools as stt
# import scipy.signal as sgn
import statsmodels.api as sm
# from statsmodels.sandbox.regression.predstd import wls_prediction_std
# from matplotlib.mlab import PCA
from collections import defaultdict

#------------------------------------------------
'''Some time length'''
night_len = int(4*3600*2.5)
mor_len = int(4*3600*2.25)
aftn_len = int(4*3600*1.5)
day_len = night_len + mor_len + aftn_len + 4

#-----------------------------------------------
'''add columns'''
def AddCol(df):
    vol = df.ix[:, 'volume'].diff()
    # this addition is for the convenience of Log y scale plot
    # vol +=1
    vol = vol.rename('vol_diff')

    openint = df.ix[:, 'openInterest'].diff()
    # this addition is for the convenience of Log y scale plot
    # openint += 1
    openint = openint.rename('openInt_diff')

    mid = (df.ix[:, 'askPrc_0'] + df.ix[:, 'bidPrc_0']) / 2.
    mid = mid.rename('midPrc')
    ret = df.join([vol, openint, mid])
    return ret


# -------------------------------------------------
def ForwardDiff(df, n=1):
    """Calculate the difference of value after n rows.

    Parameters
    ----------
    df : pandas DataFrame
    n : int

    Returns
    -------
    ret : DataFrame.
    
    """
    ret = df.diff(periods=n)
    ret = ret.shift(periods= -1 * n)
    ret = ret.dropna()
    return ret

def CutHighVar(df, length=200):
    '''
    Purpose: Cut a small period after opening in the morning and at night.
    Because this time range, the var of price is high, which harmd our model.

    df: pd.DataFrame or pd.Series. With datetime index
    length: int. the length you want to cut, counted in ticks. Cannot be larger than 240
    '''
    ret = df
    bool_arr1 = np.logical_or(ret.index.hour == 21, ret.index.hour == 9)

    bool_arr = np.logical_and.reduce([bool_arr1, 
        ret.index.minute == 0, 
        ret.index.second <= int(length//4) - 1])

    ret = ret[np.logical_not(bool_arr)]
    return ret

def CutTail(df, length=60):
    '''
    Purpose: Cut a small period before market close.

    df: pd.DataFrame or pd.Series. With datetime index
    length: int. the length you want to cut, counted in ticks. Cannot be larger than 240
    '''
    ret = df
    last_boolean1 = np.logical_and.reduce(
                                          [ret.index.hour == 14, 
                                           ret.index.minute == 59, 
                                           ret.index.second >= 60 - int(length//4)])  
    # this is the last tick
    last_boolean2 = ret.index.hour == 15
    ret = ret[np.logical_not(np.logical_or(last_boolean1, last_boolean2))]
    return ret

def DayChangeNum(ser, distance=7):
    '''
    ser is price move series after process.
    distance counting in hours
    '''
    h = ser.index.hour
    h_diff = np.diff(h)
    h_diff = np.insert(h_diff, 1, 0)
    ret = np.where(np.abs(h_diff) > distance)[0]
    return ret

# def NormPriceMove(ser, daychgnum):
#     ret = ser.copy()
#     for i in range(len(daychgnum) - 1):
#         mysamp = ret.iloc[daychgnum[i]: daychgnum[i+1]]
#         #print mysamp
#         mystd = mysamp.std()
#         print mystd
#         ret.iloc[daychgnum[i]: daychgnum[i+1]] /= mystd
#     return ret

def CuthlLimit(df, forward=60, backward=100, how='all', depth=0):
    """Cut those reach high low Limit, including an extended length around them

    Parameters
    ----------
    df : Original DataFrame including all level quote infomation
    forward : forward_ticks of price move
    backward : sample length needed to generate an indicator
    how : only consider highLimit, lowLimit or allLimit
    depth : consider price which level quote reach high low Limit

    Returns
    -------
    ret : selected boolean array

    """
    extend_len = 2 * max([forward, backward]) + 1
    s1 = 'bidQty_' + str(depth)
    s2 = 'askQty_' + str(depth)
    if how == 'all':
        arr1 = df.ix[:, s1] == 0
        arr2 = df[s2] == 0
        bool_arr = np.logical_or(arr1, arr2)
        #bool_arr = np.logical_or(df[s1] == 0, df[s2] == 0)
    elif how == 'bid':
        bool_arr = (df[s1] == 0)
    elif how == 'ask':
        bool_arr = (df[s2] == 0)
    else:
        print 'ERROR!'
    float_arr = bool_arr.astype(float)
    float_arr_diffusion = pd.Series(data=float_arr).rolling(window=extend_len, center=True).mean()
    dicard_arr = float_arr_diffusion.fillna(value=1.).astype(bool)
    return np.logical_not(dicard_arr)

def GiveMePM(df, nforward=60, nbackward=100, lim=[0, 30], cutdepth=0, norm=False, high_var_length=200):
    """from original DataFrame calculate price move Series,
        including CutTail and CutHighVar.

    Parameters
    ----------
    df : the Original DataFrame.
    forward : forward_ticks of price move
    backward : sample length needed to generate an indicator
    n : forward_ticks
    lim : can be like (0, 20), counting in days, or an int array of index.
    norm : if True, normalize the price move using every day std.

    Returns
    -------
    ret : price move series.

    """
    global day_len
    if len(lim) == 2:
        samp = df.ix[day_len*lim[0]: day_len*lim[1], 'midPrc']
    else:
        samp = df.ix[lim, 'midPrc']
    #print 'samp'
    ret = ForwardDiff(samp, nforward)
    #print 'ForwardDiff'
    # ret = CuthlLimit(ret, how='all', depth=cutdepth).loc[:, 'midPrc']
    # #print 'CuthlLimit'
    ret = CutTail(ret, nforward)
    #print 'CutTail'
    cut_head_length = max([high_var_length, nbackward])
    ret = CutHighVar(ret, length=cut_head_length)
    #print 'CutHighVar'
    # if norm:
    #     ret_daychangenum = DayChangeNum(ret)
    #     ret = NormPriceMove(ret, ret_daychangenum)
    selected_arr = CuthlLimit(df, forward=nforward, backward=nbackward, how='all', depth=cutdepth)
    return ret[selected_arr].dropna()


def GiveMeIndex(arri, arro):
    '''
    Generate integer index
    arr is a two dim ndarray, with each element being a time range(counting in days).
    '''
    global day_len
    index_in = list()
    for k in arri:
        index_in = index_in + list(range(day_len * k[0], day_len * k[1]))
    index_out = list()
    for k in arro:
        index_out = index_out + list(range(day_len * k[0], day_len * k[1]))
    return index_in, index_out
