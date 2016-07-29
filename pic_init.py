import pandas as pd
import numpy as np
from ols_module import *
def Corr2D(x, y):
    """Compute coefficient of correlation of two vector
    
    Parameters
    ----------
    x, y are ndarray
    
    Returns
    -------
    scalar. coef. of corr.
    
    """
    return np.corrcoef(x, y)[0, 1]

def PowerTransDisPlot(ser, method='pow', n=1., ret=False):
    if method == 'pow':
        ser_trans = np.sign(ser) * np.power(np.abs(ser), n)
    elif method == 'log':
        ser_trans = np.sign(ser) * np.log(np.abs(ser) + 1)
    elif method == 'powshrink':
        ser_trans = np.sign(ser) * (np.power(np.abs(ser) + 1, n) - 1)
    else:
        print 'ERROR! in PowerTransDisPlot()'
    if ret:
        return ser_trans
    else:
        sns.distplot(ser_trans)
def Splitit(x_full, y_full, y_out):
    bool_arr = y_full.index.map(lambda x: x not in y_out.index)
    select_index = (y_full.index)[bool_arr]
    return x_full.ix[select_index], x_full.ix[y_out.index], y_full.ix[select_index]

def MyRgrs(xi, xo, yi, yo, model=None, align=False):
    '''use yi and yout 's index to align. x is of full length.

    Parameters
    ----------
    model : sklearn clf

    Returns
    -------
    res : regression result
    
    '''
    if align:
        xi = xi.ix[yi.index]
        xo = xo.ix[yo.index]
    if not xi.ndim > 1:
        xi = xi.reshape(-1, 1)
        xo = xo.reshape(-1, 1)
    # if norm:
    #     xi = preprocessing.normalize(xi, norm='l2')
    res = model.fit(xi, yi)
    rsq_in = res.score(xi, yi)
    print ('rsq_in: %f' % rsq_in)
    rsq_out = res.score(xo, yo)
    print ('rsq_out: %f' % rsq_out)
    return res, rsq_in, rsq_out

def PropMatrix(df):
    ols_mat = df.apply(lambda col: Myols(col, yin, yout, False), axis=0).T
    corr_xy = df.apply(lambda col: Corr2D(col, y0), axis=0)
    corr_xy.name = 'corr with Y'
    ols_mat = ols_mat.join(corr_xy)
    dscrb_mat = df.apply(lambda x: x.describe(), axis=0).T
    x0_prop = ols_mat.join(dscrb_mat.drop(labels='count', axis=1))
    return x0_prop

def AbnormalFeature():
    unique_len_arr = []
    for i in range(80):
        n = len(np.unique(x0.iloc[:, i]))
        unique_len_arr.append(n)
        print '%d' %i
    unique_len_arr = np.array(unique_len_arr)
    unique_percent = unique_len_arr *1./ len(x0)
    sns.distplot(unique_percent, bins=10, kde=False, rug=True)
    unique_percent = pd.Series(data=unique_percent, index=x0.columns)
    abnormal_index = np.array([6, 7, 8, 27, 33, 34, 35, 58, 59, 60, 71, 73, 74, 75, 76, 77, 78, 79])
    abnormal_index = 'x' + abnormal_index.astype(str).astype(object)
    print abnormal_index
    normal_index = [x for x in x0.columns if x not in abnormal_index]
    normal_index = np.array(normal_index).astype(object)
    normal_index

def SaveDistPlot(df):
    for i in range(80):
        fig = plt.figure(figsize=(10,8))
        ax1 = fig.add_subplot(111)
        sns.distplot(df.iloc[:, i], ax=ax1)
        plt.title('x%d'%i)
        #plt.show()
        plt.savefig('./distplots/distplot_x%d'%i)
        plt.close()
        print '%d' %i

def PlotFit(res, xin, xout, yin, yout, n):
    plt.figure(figsize=(10,8))
    plt.scatter(xin.ix[:, n], 
                yin.ix[:, n], c='k', label='data', facecolors='none')
    plt.hold('on')
    plt.scatter(xin.ix[:, n], 
                res.predict(xin), edgecolors='r', label='Linear model', facecolors='none')
    plt.figure(figsize=(10,8))
    plt.scatter(xout.ix[:, n], 
                yout.ix[:, n], c='k', label='data', facecolors='none')
    plt.hold('on')
    plt.scatter(xout.ix[:, n], 
                res.predict(xout), edgecolors='r', label='Linear model', facecolors='none')

def GridPlot(res, range_arr, rsq_i_arr, rsq_o_arr, nonzero_arr, coef_arr):
    fig = plt.figure(figsize=(10,8))
    # Plot R-square of insample and out sample
    ax1 = fig.add_subplot(211)
    ax1.plot(range_arr, rsq_i_arr, label='R_in', color='orange', marker='x')
    ax1.plot(range_arr, rsq_o_arr, label='R_out', color='red', marker='x')
    ax1.set_ylim([0., .12])
    ax1.legend(loc='upper left')
    ax1.set_title('Left is Rsquare, Right is non-zero coef. percentage')
    # Plot percentage of Non-zero coefficients
    ax2 = ax1.twinx()
    ax2.plot(range_arr, nonzero_arr, label='Nonzero_len', color='blue', marker='x')
    ax2.set_ylim([0., 1.])
    ax2.legend(loc='upper right')
    # plot coefficients of Xs
    ax3 = fig.add_subplot(212)
    ax3.plot(range_arr, coef_arr)
    ax3.set_title(r'value of coef. of $X_3$')

def GridContour(x, y, z):
    z = np.array(z)
    Z = z.reshape(len(y), len(x))
    X, Y = np.meshgrid(x, y)
    #plt.figure()
    CS = plt.contourf(X, Y, Z, np.arange(0., 0.08, 0.001), cmap=plt.cm.bone, origin='lower')
    CS2 = plt.contour(CS, levels=CS.levels[::5],
                  colors='r',
                  origin='lower',
                  hold='on')
    cbar = plt.colorbar(CS)
    cbar.add_lines(CS2)
    plt.show()
    return X, Y, Z

# x0_standardized = pd.DataFrame(index=x0.index, columns=x0.columns, data=preprocessing.scale(x0, axis=0, with_mean=False))
pic = pd.read_pickle('/home/bingnan/ecworkspace/HFT1/sample.pic')

x0 = pic.iloc[:, :-1].copy()
y0 = pic.iloc[:, -1].copy()

yout = y0[int(len(y0)*0.6):]
xin, xout, yin = Splitit(x0, y0, yout)

x0_prop = PropMatrix(x0)


# for myepsilon in np.arange(.2, .5, .5):
#     for myc in np.arange(.1, .9, .9):
#         for mygamma in np.arange(.01, .05, .05):
#             mod = svm.SVR(kernel='rbf', epsilon=myepsilon, C=myc, gamma='auto', cache_size=12000., degree=3)
#             #mod = svm.LinearSVR(epsilon='linear', C=.00002, loss='epsilon_insensitive')
#             print ('\n\n===========$\epsilon$: %f, $C$: %f, $\gamma$: %f===================' % (myepsilon, myc, mygamma))
#             res = MyRgrs(x0_standardized
#                          , yin2, yout2, mod)

