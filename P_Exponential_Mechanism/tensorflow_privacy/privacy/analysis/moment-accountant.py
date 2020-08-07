import math
import sys
import numpy as np
import scipy.integrate as integrate
import scipy.stats
from mpmath import mp



def _to_np_float64(v):
  if math.isnan(v) or math.isinf(v):
    return np.inf
  return np.float64(v)


######################
# FLOAT64 ARITHMETIC #
######################
#函数的形式定义
def pdf_general_mp(x, b, mean):
  return (mp.mpf(1.) / integral__) * mp.exp(-np.abs(x-mean)**p/b)

#函数的无界积分
def integral_inf_mp(fn):
  integral, _ = mp.quad(fn, [-mp.inf, mp.inf], error=True)
  return integral


#函数的有界积分
def integral_bounded_mp(fn, lb, ub):
  integral, _ = mp.quad(fn, [lb, ub], error=True)
  return integral

#输出三个分布
def distributions_mp(b, q):
  mu0 = lambda y: pdf_general_mp(y, b=b, mean=mp.mpf(0))
  mu1 = lambda y: pdf_general_mp(y, b=b, mean=mp.mpf(1))
  mu = lambda y: (1 - q) * mu0(y) + q * mu1(y)
  return mu0, mu1, mu

#输出一次迭代的α
def compute_a_mp(b, q, lmbd, verbose=True):
  lmbd_int = int(math.ceil(lmbd))
  if lmbd_int == 0:
    return 1.0

  mu0, mu1, mu = distributions_mp(b, q)
  a_lambda_fn = lambda z: mu(z) * (mu(z) / mu0(z)) ** lmbd_int
  a_lambda_first_term_fn = lambda z: mu0(z) * (mu(z) / mu0(z)) ** lmbd_int
  a_lambda_second_term_fn = lambda z: mu1(z) * (mu(z) / mu0(z)) ** lmbd_int

  a_lambda = integral_inf_mp(a_lambda_fn)
  a_lambda_first_term = integral_inf_mp(a_lambda_first_term_fn)
  a_lambda_second_term = integral_inf_mp(a_lambda_second_term_fn)

  if verbose:
    print ("A: by numerical integration {} = {} + {}".format(
        a_lambda,
        (1 - q) * a_lambda_first_term,
        q * a_lambda_second_term))

  return _to_np_float64(a_lambda)

#根据每一步计算得到的log-moment和epsilon计算出δ
def _compute_delta(log_moments, eps):
  min_delta = 1.0
  for moment_order, log_moment in log_moments:
    if moment_order == 0:
      continue
    if math.isinf(log_moment) or math.isnan(log_moment):
      sys.stderr.write("The %d-th order is inf or Nan\n" % moment_order)
      continue
    if log_moment < moment_order * eps:
      min_delta = min(min_delta,
                      math.exp(log_moment - moment_order * eps))
  return min_delta

#根据每一步计算得到的log-moment和δ计算出epsilon
def _compute_eps(log_moments, delta):
  array=[]
  min_eps = float("inf")
  for moment_order, log_moment in log_moments:
    if moment_order == 0:
      continue
    if math.isinf(log_moment) or math.isnan(log_moment):
      sys.stderr.write("The %d-th order is inf or Nan\n" % moment_order)
      continue
    array.append((moment_order,(log_moment - math.log(delta)) / moment_order))
    min_eps = min(min_eps, (log_moment - math.log(delta)) / moment_order)
  print (array)
  return min_eps,array


def compute_log_moment(q, b, steps, lmbd, verify=False, verbose=False):
  moment = compute_a_mp(b, q, lmbd, verbose=verbose)
  if np.isinf(moment):
    return np.inf
  else:
    return np.log(moment) * steps


def get_privacy_spent(log_moments, target_eps=None, target_delta=None):
  """Compute delta (or eps) for given eps (or delta) from log moments.
  Args:
    log_moments: array of (moment_order, log_moment) pairs.
    target_eps: if not None, the epsilon for which we would like to compute
      corresponding delta value.
    target_delta: if not None, the delta for which we would like to compute
      corresponding epsilon value. Exactly one of target_eps and target_delta
      is None.
  Returns:
    eps, delta pair
  """
  assert (target_eps is None) ^ (target_delta is None)
  assert not ((target_eps is None) and (target_delta is None))
  if target_eps is not None:
    return (target_eps, _compute_delta(log_moments, target_eps))
  else:
    return (_compute_eps(log_moments, target_delta), target_delta)


if __name__ == '__main__':
  # P=[2.0+0.1*x for x in range(11)]

# B=[34.5000000000000,50.4690574808896,74.1372651027921,109.343102075613,159.103073582021,236.319021998160,352.299854402153,517.043176420742,775.844878124437,1168.10790884662,1728.25000000000]
  p=2.0
  b=277
  Eps=[]
  q=1/250
  T=1
  pdf_general = lambda z: mp.exp(-np.abs(z) ** p/ b)
  integral__, _ = mp.quad(pdf_general, [-mp.inf, mp.inf], error=True)
  parameters=[[q,b,T]]
  delta=0.00001
  max_lmbd = 19
  lmbds = range(max_lmbd , max_lmbd + 1)
  log_moments = []
  for lmbd in lmbds:
    log_moment = 0
    for q, b, T in parameters:
      log_moment += compute_log_moment(q, b, T, lmbd)
    log_moments.append((lmbd, log_moment))
  eps,  delta = get_privacy_spent(log_moments, target_delta=delta)
  Eps.append(eps)
  # print (order)
  # print ('p,b:',p,b)
  print ("eps, delta:",eps, delta)