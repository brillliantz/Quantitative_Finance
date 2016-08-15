_distance_wrap.cdist_weighted_minkowski_wrap(XA, XB, dm, p, w)

static PyObject *cdist_weighted_minkowski_wrap(PyObject *self, PyObject *args) {
  PyArrayObject *XA_, *XB_, *dm_, *w_;
  int mA, mB, n;
  double *dm;
  const double *XA, *XB, *w;
  double p;
  if (!PyArg_ParseTuple(args, "O!O!O!dO!",
			&PyArray_Type, &XA_, &PyArray_Type, &XB_, 
			&PyArray_Type, &dm_,
			&p,
			&PyArray_Type, &w_)) {
    return 0;
  }
  else {
    NPY_BEGIN_ALLOW_THREADS;
    XA = (const double*)XA_->data;
    XB = (const double*)XB_->data;
    w = (const double*)w_->data;
    dm = (double*)dm_->data;
    mA = XA_->dimensions[0];
    mB = XB_->dimensions[0];
    n = XA_->dimensions[1];
    cdist_weighted_minkowski(XA, XB, dm, mA, mB, n, p, w);
    NPY_END_ALLOW_THREADS;
  }
  return Py_BuildValue("d", 0.0);
}

cdist_weighted_minkowski(const double *XA, const double *XB, double *dm,
                         npy_intp mA, npy_intp mB, npy_intp n, double p,
                         const double *w)
{
    npy_intp i, j;
    const double *u, *v;

    for (i = 0; i < mA; i++) {
        for (j = 0; j < mB; j++, dm++) {
            u = XA + (n * i);
            v = XB + (n * j);
            *dm = weighted_minkowski_distance(u, v, n, p, w);
        }
    }
}

static NPY_INLINE double
weighted_minkowski_distance(const double *u, const double *v, npy_intp n,
                            double p, const double *w)
{
    npy_intp i = 0;
    double s = 0.0, d;
    for (i = 0; i < n; i++) {
        d = fabs(u[i] - v[i]) * w[i];
        s += pow(d, p);
    }
    return pow(s, 1.0 / p);
}