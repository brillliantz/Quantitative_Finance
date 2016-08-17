import pandas as pd
import numpy as np
from ols_module import *

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import pairwise
from sklearn import metrics
from sklearn import svm
from sklearn import linear_model
from sklearn import preprocessing

from scipy.spatial import distance

import functools
import time as tm


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
        print 'ERROR! In PowerTransDisPlot()'
    if ret:
        return ser_trans
    else:
        sns.distplot(ser_trans)
# def Splitit(x_full, y_full, y_out):
#     bool_arr = y_full.index.map(lambda x: x not in y_out.index)
#     select_index = (y_full.index)[bool_arr]
#     return x_full.ix[select_index], x_full.ix[y_out.index], y_full.ix[select_index]
def Splitit(x_full, y_full, percent1, percent2):
    N = len(y_full)
    n1, n2 = int(N * percent1), int(N * percent2)
    in_index, out_index, test_index = y_full.index[:n1], y_full.index[n1: n2], y_full.index[n2:]
    return (x_full.ix[in_index], y_full.ix[in_index],
        x_full.ix[out_index], y_full.ix[out_index],
        x_full.ix[test_index], y_full.ix[test_index])


def svrRgr(xi, xo, yi, yo, model=None, regressor_kws={}, kernel_kws={}, align=True):
    '''use yi and yout 's index to align. x is of full length.

    Parameters
    ----------
    model : sklearn clf

    Returns
    -------
    res : regression result

    '''
    import functools
    if align: # use Y as align index
        xi = xi.ix[yi.index]
        xo = xo.ix[yo.index]
    if not xi.ndim > 1: # for univariate regression
        xi = xi.reshape(-1, 1)
        xo = xo.reshape(-1, 1)
    partial_kernel = functools.partial(kernel, **kernel_kws)
    regressor = svm.SVR(kernel=partial_kernel, **regressor_kws)
    res = regressor.fit(xi, yi)
    rsq_in = res.score(xi, yi)
    print ('rsq_in: %f' % rsq_in)
    rsq_out = res.score(xo, yo)
    print ('rsq_out: %f' % rsq_out)
    return res, rsq_in, rsq_out

from sklearn import linear_model

class myRgr(object):
    def __init__(self):
        pass
    def timeit(func):
        def timed(*args, **kwargs):
            ts = time.time()
            result = func(*args, **kwargs)
            te = time.time()
            print '%r %.2f sec' % (func.__name__, te-ts)
            return result
        return timed
    def modelGo(self, regressor, regressor_kws={} ,kernel=None, kernel_kws={}):
        """
        initialize regressor (with kernel if any).

        """
        self.xin = xin
        self.xout = xout
        if regressor == 'svr' and callable(kernel):
            self.kernel = kernel
            self.partial_kernel = functools.partial(kernel, **kernel_kws)
            self.regressor = svm.SVR(kernel=self.partial_kernel, **regressor_kws)
        elif regressor == 'Ridge':
            self.regressor = linear_model.Ridge(**regressor_kws)
        elif regressor == 'Lasso':
            self.regressor = linear_model.Lasso(**regressor_kws)
        else:
            raise ValueError('Wrong model!')
    def dataGo(self, xin, yin, xout, yout, xtest, ytest, align=True):
        """all arguments are pd.DataFrame or pd.Series type.
        """
        self.xin, self.yin = xin, yin
        self.xout, self.yout = xout, yout
        self.xtest, self.ytest = xtest, ytest
        if align:
            self.xin = self.xin.ix[self.yin.index]
            self.xout = self.xout.ix[self.yout.index]
            self.xtest = self.xtest.ix[self.ytest.index]
        if not self.xin.ndim > 1: # for univariate regression
            self.xin = self.xin.reshape(-1, 1)
            self.xout = self.xout.reshape(-1, 1)
            self.xtest = self.xtest.reshape(-1, 1)
        # self.half_life = np.log(2) / np.mean(self.partial_kernel(xin, xin)) # for gamma
    @timeit
    def fit(self):
        self.result = self.regressor.fit(self.xin, self.yin)
    @timeit
    def predict(self, x_new):
        return self.result.predict(x_new)
    @timeit
    def rsqGo(self):
        self.rsq_in = self.result.score(self.xin, self.yin)
        self.rsq_out = self.result.score(self.xout, self.yout)
        print 'rsq_in: {0:.6f} \nrsq_out: {1:.6f}'.format(self.rsq_in * 100, self.rsq_out * 100)
    @timeit
    def test(self):
        #self._t = tm.time()
        self.rsq_test = self.result.score(self.xtest, self.ytest)
        print 'rsq_test: {0:.6f} '.format(self.rsq_test * 100)
        #print '{0:.4f}s for rsq test.'.format(tm.time() - self._t)
    @timeit
    def residualGo(self):
        self._yin_predict = self.predict(self.xin)
        self.residual = self._yin_predict - self.yin


def MyRgrs(xi, xo, yi, yo, model=None, align=True):
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

def PlotFit(res, xi, xo, yi, yo, n):
    import matplotlib.pyplot as plt
    xi = xi.ix[yi.index]
    xo = xo.ix[yo.index]
    if type(res) == sm.regression.linear_model.RegressionResultsWrapper:
        xi = sm.add_constant(xi)
    plt.figure(figsize=(20,8))
    plt.scatter(xi.ix[:, n],
                yi, c='k', label='data', facecolors='none')
    plt.hold('on')
    plt.scatter(xi.ix[:, n],
                res.predict(xi), edgecolors='r', label='Linear model', facecolors='none')
    plt.figure(figsize=(20,8))
    plt.scatter(xo.ix[:, n],
                yo, c='k', label='data', facecolors='none')
    plt.hold('on')
    plt.scatter(xo.ix[:, n],
                res.predict(xo), edgecolors='r', label='Linear model', facecolors='none')

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
    CS = plt.contourf(X, Y, Z, np.arange(0.05, 0.08, 0.001), cmap=plt.cm.bone, origin='lower')
    CS2 = plt.contour(CS, levels=CS.levels[::2],
                  colors='r',
                  origin='lower',
                  hold='on')
    cbar = plt.colorbar(CS)
    cbar.add_lines(CS2)
    plt.show()
    return X, Y, Z

def KernelDist(kernel1, kwargs1, kernel2=None, kwargs2=None, resample=200):
    """
    kwargs1 and kwargs2 are dictionary.

    """
    global xin_stdzd
    a, b = xin_stdzd.ix[::resample], xin_stdzd.ix[::resample]
    if kernel2:
        temp1, temp2 = kernel1(a, b, **kwargs1), kernel2(a, b, **kwargs2)
    else:
        temp1 = kernel1(a, b, **kwargs1)
    plt.figure(figsize=(16,8))
    #ax1 = fig.add_subplot(111)
    sns.distplot(temp1.ravel(), label='kernel1')
    if kernel2:
        sns.distplot(temp2.ravel(), label='kernel2')
    plt.legend()
#     return fig

def my_kernel_exp(x, y, gamma=.2, metric='eu', squared=False, w=None, full_output=False):
    """gamma is auto tuned. The argument gamma means shrink_ratio.

    """
    if y.shape[1] != x.shape[1]:
        raise ValueError('x and y shape do not match')
    dim = x.shape[1]e
    if w != None:
        x = x * w
        y = y * w
    if metric=='eu':
        f = pairwise.euclidean_distances
        dim = np.sqrt(dim)
    elif metric=='mh':
        f = pairwise.manhattan_distances
    else:
        raise ValueError('WrongMetricError')
    dis = f(x, y)
    dis *= 1./dim # normalize
    if full_output:
        print 'gamma = {0:.4f}     dim = {1:.4f}     squared = {2}'.format(gamma, dim, squared)
    if squared:
        dis += 1.
        dis = dis**2
        dis -= 1.
        # choose exp(-gamma*x^2) or exp(-gamma*x)
#         print np.log(2) / np.median(dis)
    dis *= -gamma
    np.exp(dis, dis)
    return dis

pic = pd.read_pickle('/home/bingnan/ecworkspace/HFT1/sample.pic')

x0 = pic.iloc[:, :-1].copy()
y0 = pic.iloc[:, -1].copy()

xin, yin, xout, yout, xtest, ytest = Splitit(x0, y0, 0.6, 0.8)

#----------------- comment below when calculating
x0_prop = PropMatrix(x0)
#----------------- comment above when calculating

_xin_mean = xin.mean(axis=0)
_xin_std = xin.std(axis=0)
xin_stdzd = (xin - _xin_mean) / _xin_std
xout_stdzd = (xout - _xin_mean) / _xin_std
xtest_stdzd = (xtest - _xin_mean) / _xin_std
