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
import time


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

def dict2kwargs(myinput):
    if isinstance(myinput, dict):
        res = ''
        for key, item in myinput.iteritems():
            res = res + key + '=' + str(item) + ', '
        print res
    elif isinstance(myinput, str):
        kwargs = {}
        arguments = myinput.split(', ')
        for argu in arguments:
            try:
                key, item = argu.split('=')
                kwargs[key] = item
            except:
                raise ValueError('you can only input keyword parameters!')
        print kwargs
    else:
        raise ValueError('wrong input!')

def my_kernel_rquad(dis, c):
    return  1. * c / (dis + c)

def my_kernel_exp(dis, gamma=.1):
    """gamma is auto tuned. The argument gamma means shrink_ratio.
    
    """
    # if full_output:
    #     print 'gamma = {0:.4f}     dim = {1:.4f}     squared = {2}'.format(gamma, dim, squared)
    dis *= -gamma
    np.exp(dis, dis)
    return dis

def my_distance(x, y, metric='eu', squared=False, w=None):
    """only calculate pairwise distance between matrix x and y.
    
    """
    dim = x.shape[1]
    if w is not None:
        x = x * w
        y = y * w
    if metric == 'eu':
        f = pairwise.euclidean_distances
        dim = np.sqrt(dim)
    elif metric == 'mh':
        f = pairwise.manhattan_distances
    else:
        raise ValueError('Wrong Metric !')
    dis = f(x, y)
    dis *= 1. / dim # normalize
    if squared: # choose exp(-gamma*x^2) or exp(-gamma*x)
        dis = dis**2
    return dis

def rsquare(ytrue, yhat=None, residual=None): # TODO get out of this class, cz u don't need 'self' argument
    if yhat is not None:
        print 'using yhat'
        return 1. - ((yhat - ytrue)**2).mean() / ytrue.var(ddof=0)
    elif residual is not None:
        print 'using residual'
        return 1. - (residual**2).mean() / ytrue.var(ddof=0)
    else:
        raise ValueError('When calculating Rsquare, you must input yhat or residual!')

def _timeit(func):
        def timed(*args, **kwargs):
            ts = time.time()
            result = func(*args, **kwargs)
            te = time.time()
            print '%r %.2f sec' % (func.__name__, te-ts)
            return result
        return timed

class myRgr(object):
    def __init__(self):
        """Some marker"""
        self.rzdu_in = None
        self.rzdu_out = None
        self.half_life = None
        self._dataGo_ready = False
        self._disPrep_ready = False
    
    def dataGo(self, xin, yin, xout, yout, xtest, ytest, 
                align=True, index=None, cols=None):
        """all arguments are pd.DataFrame or pd.Series type.

        """
        self.xin, self.yin = xin, yin
        self.xout, self.yout = xout, yout
        self.xtest, self.ytest = xtest, ytest
        
        if cols is not None:
            self.xin, self.xout, self.xtest = (self.yin.ix[:, cols], 
                                                self.xout.ix[:, cols], 
                                                self.xtest.ix[:, cols])
        if index is not None:
            pass
        
        if align:
            self.xin = self.xin.ix[self.yin.index]
            self.xout = self.xout.ix[self.yout.index]
            self.xtest = self.xtest.ix[self.ytest.index]
        
        if not self.xin.ndim > 1: # for univariate regression
            self.xin = self.xin.reshape(-1, 1)
            self.xout = self.xout.reshape(-1, 1)
            self.xtest = self.xtest.reshape(-1, 1)
        
        self._dataGo_ready = True

    def disPrep(self, dis_func, **dis_kws):
        """Only for self-defined kernel SVR

        Parameters
        ----------
        dis_func: its arguments must be of the form:
                    (x, y, **kwargs)

        """

        if not self._dataGo_ready:
            raise AssertionError('func. dataGo should run before func. prepare !')

        #TODO
        """pre-compute distance between in-in and in-out for performance consideration"""
        self.dis_func, self.dis_kws = dis_func, dis_kws
        self._dis_in_in = self.dis_func(self.xin, self.xin, **self.dis_kws)
        self._dis_out_in = self.dis_func(self.xout, self.xin, **self.dis_kws)
        self._dis_kws = dis_kws
        self.median_dis = np.median(my_distance(self.xin, self.xin, **self._dis_kws)) # for gamma
        self.half_life = np.log(2) / self.median_dis

        self._disPrep_ready = True
    
    # def exp_kernel(self, )
    def kernelGo(self, kernel_func, **kernel_kws):
        """Initialize kernel and its kwargs, then define 
        partial_kernel (to pass to SVR regressor)

        Parameters
        ----------
        kernel_func: its arguments must be of the form:
                    (x, y, **kwargs)

        """
        #TODO
        self.kernel, self.kernel_kws = kernel_func, kernel_kws
        if callable(self.kernel):
            def partial_kernel(x, y):
                """y will always be self.xin, so whether we 
                use pre-computed matrix is depend on x"""
                if x.shape == self.xin.shape:
                    if np.abs((x - self.xin).iloc[0, :]).sum() < 1e-10:
                        print 'in-in'
                        dis = self._dis_in_in.copy()
                    else:
                        print 're-calc!!!'
                        dis = self.dis_func(x, y, **self.dis_kws)
                elif x.shape == self.xout.shape:
                    if ((x - self.xout).iloc[0, :]).sum() < 1e-10:
                        print 'out-in'
                        dis = self._dis_out_in.copy()
                    else:
                        print 're-calc!!!'
                        dis = self.dis_func(x, y, **self.dis_kws)
                else:
                    print 're-calc!!!'
                    dis = self.dis_func(x, y, **self.dis_kws)
                return self.kernel(dis, **self.kernel_kws)
            self.partial_kernel = partial_kernel
        
        elif isinstance(self.kernel, str):
            self.partial_kernel = self.kernel
        else:
            raise ValueError('Wrong kernel passed !')
    
    def regressorGo(self, regressor, **regressor_kws):
        """
        initialize regressor (with kernel if any).

        """
        # if not self._disPrep_ready:
        #     raise AssertionError('func. regressorGo should run before func. disPrep !')
        if regressor == 'svr':
            self.regressor = svm.SVR(kernel=self.partial_kernel, **regressor_kws)
        elif regressor == 'Ridge':
            self.regressor = linear_model.Ridge(**regressor_kws)
        elif regressor == 'Lasso':
            self.regressor = linear_model.Lasso(**regressor_kws)
        else:
            raise ValueError('input regressor not recognized !')
    
    @_timeit
    def fit(self):
        self.result = self.regressor.fit(self.xin, self.yin)
    
    @_timeit
    def predict(self, x_new):
        return self.result.predict(x_new)
    
    @_timeit
    def residualGo(self):
        self._yin_predict = self.predict(self.xin)
        self._yout_predict = self.predict(self.xout)
        self.rzdu_in = self.yin - self._yin_predict
        self.rzdu_out = self.yout - self._yout_predict
    
    @_timeit
    def rsqGo(self):
        """Calculate insample and outsample rsquare"""
        # if self.rzdu_in is None or self.rzdu_out is None:
        #     self.residualGo()
        self.residualGo()
        self.rsq_in = rsquare(self.yin, residual=self.rzdu_in)
        self.rsq_out = rsquare(self.yout, residual=self.rzdu_out)
        print '\t\t\t\t\t  ---rsq_in: {0:.6f}\n\t\t\t\t\t ---rsq_out: {1:.6f}'.format(
            self.rsq_in, self.rsq_out)
    
    @_timeit
    def test(self):
        self.rsq_test = self.result.score(self.xtest, self.ytest)
        print '\t\t\t\t\t---rsq_test: {0:.6f} '.format(self.rsq_test)
    
    """
    ----- wGo and decision_func are only for linear SVR ------

    def wGo(self):
        self.dual_coef = np.zeros_like(self.yin)
        count = 0
        dual_coef_only_sv = self.result.dual_coef_.ravel()
        for i in range(len(self.yin)):
            if i in self.result.support_:
                self.dual_coef[i] = dual_coef_only_sv[count]
                count += 1
        self.w = np.dot(self.dual_coef, self.xin)
    
    def decision_func(self):
        try:
            self.w + 1
        except:
            self.wGo()
        def wrapper(x):
            return np.dot(x, self.w) + self.result.intercept_
        return wrapper

    """

    def plotFit(self, n):
        """Plot ypre to x_n and yture to x_n, in-and-out sample"""
        plt.figure(figsize=(20,8))
        plt.scatter(self.xin.ix[:, n], self.yin, 
            edgecolors='k', facecolors='none', label='yin_true')
        plt.hold('on')
        plt.scatter(self.xin.ix[:, n], self._yin_predict, 
            edgecolors='r', facecolors='none', label='yin_predict')
        plt.legend()
        plt.figure(figsize=(20,8))
        plt.scatter(self.xout.ix[:, n], self.yout, 
            edgecolors='k', facecolors='none', label='yout_true')
        plt.hold('on')
        plt.scatter(self.xout.ix[:, n], self._yout_predict, 
            edgecolors='r', label='yout_predict', facecolors='none')
        plt.legend()



# def MyRgrs(xi, xo, yi, yo, model=None, align=True):
#     '''use yi and yout 's index to align. x is of full length.

#     Parameters
#     ----------
#     model : sklearn clf

#     Returns
#     -------
#     res : regression result

#     '''
#     if align:
#         xi = xi.ix[yi.index]
#         xo = xo.ix[yo.index]
#     if not xi.ndim > 1:
#         xi = xi.reshape(-1, 1)
#         xo = xo.reshape(-1, 1)
#     # if norm:
#     #     xi = preprocessing.normalize(xi, norm='l2')
#     res = model.fit(xi, yi)
#     rsq_in = res.score(xi, yi)
#     print ('rsq_in: %f' % rsq_in)
#     rsq_out = res.score(xo, yo)
#     print ('rsq_out: %f' % rsq_out)
#     return res, rsq_in, rsq_out


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
