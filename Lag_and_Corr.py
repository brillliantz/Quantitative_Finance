import numpy as np
def Lag_and_Corr(x, y, max_lag=50):
    # max_lag = 50
    mylag_arr = range(-max_lag, max_lag, 1)
    corr_arr = np.zeros_like(mylag_arr)
    for i, mylag in enumerate(mylag_arr):
        x_lag = x.shift(n=mylag)
        x_with_y = np.vstack(x_lag, y)
        x_with_y.dropna(inplace=True)
        corr = np.corrcoef(x_with_y)[0, 1]
        corr_arr[i] = corr
    return (mylag_arr, corr_arr)

