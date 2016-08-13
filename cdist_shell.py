import warnings
import numpy as np

from scipy._lib.six import callable, string_types
from scipy._lib.six import xrange

from scipy.spatial import _distance_wrap
from scipy.linalg import norm



_SIMPLE_CDIST = {}
def _copy_array_if_base_present(a):
    """
    Copies the array if its base points to a parent array.
    """
    if a.base is not None:
        return a.copy()
    elif np.issubsctype(a, np.float32):
        return np.array(a, dtype=np.double)
    else:
        return a

def _convert_to_double(X):
    if X.dtype != np.double:
        X = X.astype(np.double)
    if not X.flags.contiguous:
        X = X.copy()
    return X

def cdist(XA, XB, metric='euclidean', p=2, V=None, VI=None, w=None):


    XA = np.asarray(XA, order='c')
    XB = np.asarray(XB, order='c')

    # The C code doesn't do striding.
    XA = _copy_array_if_base_present(_convert_to_double(XA))
    XB = _copy_array_if_base_present(_convert_to_double(XB))

    s = XA.shape
    sB = XB.shape

    if len(s) != 2:
        raise ValueError('XA must be a 2-dimensional array.')
    if len(sB) != 2:
        raise ValueError('XB must be a 2-dimensional array.')
    if s[1] != sB[1]:
        raise ValueError('XA and XB must have the same number of columns '
                         '(i.e. feature dimension.)')

    mA = s[0]
    mB = sB[0]
    n = s[1]
    dm = np.zeros((mA, mB), dtype=np.double)

    if callable(metric):
        if metric == minkowski:
            for i in xrange(0, mA):
                for j in xrange(0, mB):
                    dm[i, j] = minkowski(XA[i, :], XB[j, :], p)
        elif metric == wminkowski:
            for i in xrange(0, mA):
                for j in xrange(0, mB):
                    dm[i, j] = wminkowski(XA[i, :], XB[j, :], p, w)
        elif metric == seuclidean:
            for i in xrange(0, mA):
                for j in xrange(0, mB):
                    dm[i, j] = seuclidean(XA[i, :], XB[j, :], V)
        elif metric == mahalanobis:
            for i in xrange(0, mA):
                for j in xrange(0, mB):
                    dm[i, j] = mahalanobis(XA[i, :], XB[j, :], V)
        else:
            for i in xrange(0, mA):
                for j in xrange(0, mB):
                    dm[i, j] = metric(XA[i, :], XB[j, :])
    elif isinstance(metric, string_types):
        mstr = metric.lower()

        try:
            validate, cdist_fn = _SIMPLE_CDIST[mstr]
            XA = validate(XA)
            XB = validate(XB)
            cdist_fn(XA, XB, dm)
            return dm
        except KeyError:
            pass

        if mstr in ['hamming', 'hamm', 'ha', 'h']:
            if XA.dtype == bool:
                XA = _convert_to_bool(XA)
                XB = _convert_to_bool(XB)
                _distance_wrap.cdist_hamming_bool_wrap(XA, XB, dm)
            else:
                XA = _convert_to_double(XA)
                XB = _convert_to_double(XB)
                _distance_wrap.cdist_hamming_wrap(XA, XB, dm)
        elif mstr in ['jaccard', 'jacc', 'ja', 'j']:
            if XA.dtype == bool:
                XA = _convert_to_bool(XA)
                XB = _convert_to_bool(XB)
                _distance_wrap.cdist_jaccard_bool_wrap(XA, XB, dm)
            else:
                XA = _convert_to_double(XA)
                XB = _convert_to_double(XB)
                _distance_wrap.cdist_jaccard_wrap(XA, XB, dm)
        elif mstr in ['minkowski', 'mi', 'm', 'pnorm']:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            _distance_wrap.cdist_minkowski_wrap(XA, XB, dm, p)
        elif mstr in ['wminkowski', 'wmi', 'wm', 'wpnorm']:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            w = _convert_to_double(w)
            _distance_wrap.cdist_weighted_minkowski_wrap(XA, XB, dm, p, w)
        elif mstr in ['seuclidean', 'se', 's']:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            if V is not None:
                V = np.asarray(V, order='c')
                if V.dtype != np.double:
                    raise TypeError('Variance vector V must contain doubles.')
                if len(V.shape) != 1:
                    raise ValueError('Variance vector V must be '
                                     'one-dimensional.')
                if V.shape[0] != n:
                    raise ValueError('Variance vector V must be of the same '
                                     'dimension as the vectors on which the '
                                     'distances are computed.')
                # The C code doesn't do striding.
                VV = _copy_array_if_base_present(_convert_to_double(V))
            else:
                VV = np.var(np.vstack([XA, XB]), axis=0, ddof=1)
            _distance_wrap.cdist_seuclidean_wrap(XA, XB, VV, dm)
        elif mstr in ['cosine', 'cos']:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            _cosine_cdist(XA, XB, dm)
        elif mstr in ['correlation', 'co']:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            XA -= XA.mean(axis=1)[:, np.newaxis]
            XB -= XB.mean(axis=1)[:, np.newaxis]
            _cosine_cdist(XA, XB, dm)
        elif mstr in ['mahalanobis', 'mahal', 'mah']:
            XA = _convert_to_double(XA)
            XB = _convert_to_double(XB)
            if VI is not None:
                VI = _convert_to_double(np.asarray(VI, order='c'))
                VI = _copy_array_if_base_present(VI)
            else:
                m = mA + mB
                if m <= n:
                    # There are fewer observations than the dimension of
                    # the observations.
                    raise ValueError("The number of observations (%d) is too "
                                     "small; the covariance matrix is "
                                     "singular. For observations with %d "
                                     "dimensions, at least %d observations "
                                     "are required." % (m, n, n + 1))
                X = np.vstack([XA, XB])
                V = np.atleast_2d(np.cov(X.T))
                del X
                VI = np.linalg.inv(V).T.copy()
            # (u-v)V^(-1)(u-v)^T
            _distance_wrap.cdist_mahalanobis_wrap(XA, XB, VI, dm)
        elif metric == 'test_euclidean':
            dm = cdist(XA, XB, euclidean)
        elif metric == 'test_seuclidean':
            if V is None:
                V = np.var(np.vstack([XA, XB]), axis=0, ddof=1)
            else:
                V = np.asarray(V, order='c')
            dm = cdist(XA, XB, lambda u, v: seuclidean(u, v, V))
        elif metric == 'test_sqeuclidean':
            dm = cdist(XA, XB, lambda u, v: sqeuclidean(u, v))
        elif metric == 'test_braycurtis':
            dm = cdist(XA, XB, braycurtis)
        elif metric == 'test_mahalanobis':
            if VI is None:
                X = np.vstack([XA, XB])
                V = np.cov(X.T)
                VI = np.linalg.inv(V)
                X = None
                del X
            else:
                VI = np.asarray(VI, order='c')
            VI = _copy_array_if_base_present(VI)
            # (u-v)V^(-1)(u-v)^T
            dm = cdist(XA, XB, (lambda u, v: mahalanobis(u, v, VI)))
        elif metric == 'test_canberra':
            dm = cdist(XA, XB, canberra)
        elif metric == 'test_cityblock':
            dm = cdist(XA, XB, cityblock)
        elif metric == 'test_minkowski':
            dm = cdist(XA, XB, minkowski, p=p)
        elif metric == 'test_wminkowski':
            dm = cdist(XA, XB, wminkowski, p=p, w=w)
        elif metric == 'test_correlation':
            dm = cdist(XA, XB, correlation)
        elif metric == 'test_hamming':
            dm = cdist(XA, XB, hamming)
        elif metric == 'test_jaccard':
            dm = cdist(XA, XB, jaccard)
        elif metric == 'test_chebyshev' or metric == 'test_chebychev':
            dm = cdist(XA, XB, chebyshev)
        elif metric == 'test_yule':
            dm = cdist(XA, XB, yule)
        elif metric == 'test_matching':
            dm = cdist(XA, XB, matching)
        elif metric == 'test_dice':
            dm = cdist(XA, XB, dice)
        elif metric == 'test_kulsinski':
            dm = cdist(XA, XB, kulsinski)
        elif metric == 'test_rogerstanimoto':
            dm = cdist(XA, XB, rogerstanimoto)
        elif metric == 'test_russellrao':
            dm = cdist(XA, XB, russellrao)
        elif metric == 'test_sokalsneath':
            dm = cdist(XA, XB, sokalsneath)
        elif metric == 'test_sokalmichener':
            dm = cdist(XA, XB, sokalmichener)
        else:
            raise ValueError('Unknown Distance Metric: %s' % mstr)
    else:
        raise TypeError('2nd argument metric must be a string identifier '
                        'or a function.')
    return dm
