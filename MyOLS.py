import statsmodels.api as sm
import matplotlib.pyplot as plt
def myols(ser, pm, norm=False):
    '''
    ser is indicator Series
    pm is Price move Series
    sm is satatsmodel module
    this function also automatically align index of df and pm
    '''
    global sm
    ser = ser[pm.index]
    ser = ser.dropna()
    if norm:
        ser = (ser - ser.mean()) / ser.std()
    X = sm.add_constant(ser)
    Y = pm[X.index]
    model = sm.OLS(Y, X)
    ret = model.fit()
    return ret

def Rsquare(y, yhat):
    # ret = 1 - (y-yhat).var() / y.var()
    ret = 1 - ((y-yhat)**2).mean() / y.var()
    return ret

def PredictedRsquare(res, xnew, pm):
    '''
    pm: outsample price move Series
    xnew: indicator Series (or DataFrame)
    res: insample regression results (comes from statsmodel's model.fit() )
    '''
    # first we need to align xnew with outsample
    xnew = xnew[pm.index]
    xnew = xnew.dropna()
    y = pm[xnew.index]
    
    xnew = sm.add_constant(xnew)
    ynew = res.predict(xnew)
    rsq = Rsquare(y, ynew)
    return ynew, rsq

def PlotFit(fitres):
    fig, ax = plt.subplots()
    fig = sm.graphics.plot_fit(fitres, fitres.model.exog_names[1], ax=ax)
    fig.show()


