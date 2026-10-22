import math
import numpy as np
from numba import njit, prange

@njit
def interpolate_sample(x, xp, fp, method=1):
    """
    Interpolate a sample `fp` over domain `xp` at point `x`.

    Inputs:
    -------
    x : float
        The x-coordinate at which to evaluate the interpolated value
    xp: np.ndarray (float)
        The x-coordinates of the data points
    fp: np.ndarray (float)
        The y-coordinates of the data points
    method: int [0 or 1]
        Use nearest neighbor (0) or linear (1) interpolation.
    """
    n = xp.shape[0]
    m = fp.shape[1]
    ix = np.searchsorted(xp, x)
    if (ix == 0):
        result = fp[0]
    elif (ix >= n):
        result = fp[n - 1]
    else:
        dx_0 = x - xp[ix - 1]
        dx_1 = xp[ix] - x
        if method == 1:
            frac = dx_0 / (dx_0 + dx_1)
            result = (1 - frac) * fp[ix - 1] + (frac) * fp[ix]
        elif method == 0:
            if abs(dx_0) <= abs(dx_1):
                result = fp[ix - 1]
            else:
                result = fp[ix]
    return result

@njit
def interpolate_samples(xs, xp, fp, method=1):
    N = len(xs)
    M = fp.shape[1]
    output = np.zeros((N, M), dtype=np.float64)
    for i in range(N):
        x = xs[i]
        y = interpolate_sample(x, xp, fp, method=method)
        output[i, :] = y
    return output

@njit
def has_same_index(known_index, dataframe_index):
    N = len(known_index)
    has_same_len = (N == len(dataframe_index))
    if has_same_len:
        for i in range(N):
            same_value = (known_index[i] == dataframe_index[i])
            if not same_value:
                return False
        return True
    return False

@njit
def _ax_bu(startnodes, endnodes, alpha, beta, chi, gamma, i_t_prev, o_t_prev, q_t_next, indegree):
    n = endnodes.size
    m = startnodes.size
    i_t_next = np.zeros(n, dtype=np.float64)
    o_t_next = np.zeros(n, dtype=np.float64)
    indegree_t = indegree.copy()
    # Simulate output
    for k in range(m):
        startnode = startnodes[k]
        endnode = endnodes[startnode]
        while(indegree_t[startnode] == 0):
            alpha_i = alpha[startnode]
            beta_i = beta[startnode]
            chi_i = chi[startnode]
            gamma_i = gamma[startnode]
            o_t_next[startnode] += (alpha_i * i_t_next[startnode]
                                    + beta_i * i_t_prev[startnode]
                                    + chi_i * o_t_prev[startnode]
                                    + gamma_i * q_t_next[startnode])
            if startnode != endnode:
                i_t_next[endnode] += o_t_next[startnode]
            indegree_t[endnode] -= 1
            startnode = endnode
            endnode = endnodes[startnode]
    return i_t_next, o_t_next

@njit # Uncomment for large networks
def _ax(startnodes, endnodes, alpha, beta, chi, i_t_prev, o_t_prev, indegree):
    n = endnodes.size
    m = startnodes.size
    i_t_next = np.zeros(n, dtype=np.float64)
    o_t_next = np.zeros(n, dtype=np.float64)
    indegree_t = indegree.copy()
    # Simulate output
    for k in range(m):
        startnode = startnodes[k]
        endnode = endnodes[startnode]
        while(indegree_t[startnode] == 0):
            alpha_i = alpha[startnode]
            beta_i = beta[startnode]
            chi_i = chi[startnode]
            o_t_next[startnode] += (alpha_i * i_t_next[startnode]
                                    + beta_i * i_t_prev[startnode]
                                    + chi_i * o_t_prev[startnode])
            if startnode != endnode:
                i_t_next[endnode] += o_t_next[startnode]
            indegree_t[endnode] -= 1
            startnode = endnode
            endnode = endnodes[startnode]
    return i_t_next, o_t_next

@njit # Uncomment for large networks
def _apply_gain(startnodes, endnodes, gain, indegree):
    n = endnodes.size
    m = startnodes.size
    i_t_next = np.zeros(n, dtype=np.float64)
    o_t_next = np.zeros(n, dtype=np.float64)
    indegree_t = indegree.copy()
    # Simulate output
    for k in range(m):
        startnode = startnodes[k]
        endnode = endnodes[startnode]
        while(indegree_t[startnode] == 0):
            o_t_next[startnode] += gain[startnode]
            if startnode != endnode:
                i_t_next[endnode] += o_t_next[startnode]
            indegree_t[endnode] -= 1
            startnode = endnode
            endnode = endnodes[startnode]
    return i_t_next, o_t_next

@njit
def numba_init_inflows(a, indices, b):
    n = len(indices)
    for i in range(n):
        k = indices[i]
        a[k] += b[i]

@njit
def _ap(P, out, startnodes, endnodes, alpha, beta, chi,
                 indegree):
    m, n = P.shape
    assert (m == n)
    for i in range(n):
        o_t_prev = P[:, i]
        i_t_prev = np.zeros(n, dtype=np.float64)
        numba_init_inflows(i_t_prev, endnodes, o_t_prev)
        _, o_t_next = _ax(startnodes, endnodes, alpha, beta, chi,
                          i_t_prev, o_t_prev, indegree)
        out[:, i] = o_t_next
    return out

@njit(parallel=True)
def _ap_par(P, out, startnodes, endnodes, alpha, beta, chi,
                 indegree):
    m, n = P.shape
    assert (m == n)
    for i in prange(n):
        o_t_prev = P[:, i]
        i_t_prev = np.zeros(n, dtype=np.float64)
        numba_init_inflows(i_t_prev, endnodes, o_t_prev)
        _, o_t_next = _ax(startnodes, endnodes, alpha, beta, chi,
                          i_t_prev, o_t_prev, indegree)
        out[:, i] = o_t_next
    return out


@njit
def _aqat(P, out, startnodes, endnodes, alpha, beta, chi,
          indegree):
    m, n = P.shape
    assert (m == n)
    for i in range(n):
        o_t_prev = P[:, i]
        i_t_prev = np.zeros(n, dtype=np.float64)
        numba_init_inflows(i_t_prev, endnodes, o_t_prev)
        _, o_t_next = _ax(startnodes, endnodes, alpha, beta, chi,
                          i_t_prev, o_t_prev, indegree)
        out[:, i] = o_t_next
    out = out.T
    for i in range(n):
        o_t_prev = out[:, i]
        i_t_prev = np.zeros(n, dtype=np.float64)
        numba_init_inflows(i_t_prev, endnodes, o_t_prev)
        _, o_t_next = _ax(startnodes, endnodes, alpha, beta, chi,
                          i_t_prev, o_t_prev, indegree)
        out[:, i] = o_t_next
    return out

@njit(parallel=True)
def _aqat_par(P, out, startnodes, endnodes, alpha, beta, chi,
              indegree):
    m, n = P.shape
    assert (m == n)
    for i in prange(n):
        o_t_prev = P[:, i]
        i_t_prev = np.zeros(n, dtype=np.float64)
        numba_init_inflows(i_t_prev, endnodes, o_t_prev)
        _, o_t_next = _ax(startnodes, endnodes, alpha, beta, chi,
                          i_t_prev, o_t_prev, indegree)
        out[:, i] = o_t_next
    out = out.T
    for i in prange(n):
        o_t_prev = out[:, i]
        i_t_prev = np.zeros(n, dtype=np.float64)
        numba_init_inflows(i_t_prev, endnodes, o_t_prev)
        _, o_t_next = _ax(startnodes, endnodes, alpha, beta, chi,
                          i_t_prev, o_t_prev, indegree)
        out[:, i] = o_t_next
    return out

@njit
def _polevl(x, coefs, N):
    # Implementation from https://github.com/dougthor42/PyErf/blob/master/pyerf/pyerf.py
    ans = 0
    power = len(coefs) - 1
    for coef in coefs:
        ans += coef * x**power
        power -= 1
    return ans

@njit
def _p1evl(x, coefs, N):
    # Implementation from https://github.com/dougthor42/PyErf/blob/master/pyerf/pyerf.py
    return _polevl(x, [1] + coefs, N)

@njit
def ndtri(y):
    # Implementation from https://github.com/dougthor42/PyErf/blob/master/pyerf/pyerf.py
    PI = math.pi
    ROOT_2PI = math.sqrt(2 * PI)
    EXP_NEG2 = math.exp(-2)
    # approximation for 0 <= abs(z - 0.5) <= 3/8
    P0 = [
        -5.99633501014107895267E1,
        9.80010754185999661536E1,
        -5.66762857469070293439E1,
        1.39312609387279679503E1,
        -1.23916583867381258016E0,
    ]

    Q0 = [
        1.95448858338141759834E0,
        4.67627912898881538453E0,
        8.63602421390890590575E1,
        -2.25462687854119370527E2,
        2.00260212380060660359E2,
        -8.20372256168333339912E1,
        1.59056225126211695515E1,
        -1.18331621121330003142E0,
    ]

    # Approximation for interval z = sqrt(-2 log y ) between 2 and 8
    # i.e., y between exp(-2) = .135 and exp(-32) = 1.27e-14.
    P1 = [
        4.05544892305962419923E0,
        3.15251094599893866154E1,
        5.71628192246421288162E1,
        4.40805073893200834700E1,
        1.46849561928858024014E1,
        2.18663306850790267539E0,
        -1.40256079171354495875E-1,
        -3.50424626827848203418E-2,
        -8.57456785154685413611E-4,
    ]

    Q1 = [
        1.57799883256466749731E1,
        4.53907635128879210584E1,
        4.13172038254672030440E1,
        1.50425385692907503408E1,
        2.50464946208309415979E0,
        -1.42182922854787788574E-1,
        -3.80806407691578277194E-2,
        -9.33259480895457427372E-4,
    ]

    # Approximation for interval z = sqrt(-2 log y ) between 8 and 64
    # i.e., y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
    P2 = [
        3.23774891776946035970E0,
        6.91522889068984211695E0,
        3.93881025292474443415E0,
        1.33303460815807542389E0,
        2.01485389549179081538E-1,
        1.23716634817820021358E-2,
        3.01581553508235416007E-4,
        2.65806974686737550832E-6,
        6.23974539184983293730E-9,
    ]

    Q2 = [
        6.02427039364742014255E0,
        3.67983563856160859403E0,
        1.37702099489081330271E0,
        2.16236993594496635890E-1,
        1.34204006088543189037E-2,
        3.28014464682127739104E-4,
        2.89247864745380683936E-6,
        6.79019408009981274425E-9,
    ]

    sign_flag = 1

    if y > (1 - EXP_NEG2):
        y = 1 - y
        sign_flag = 0

    # Shortcut case where we don't need high precision
    # between -0.135 and 0.135
    if y > EXP_NEG2:
        y -= 0.5
        y2 = y ** 2
        x = y + y * (y2 * _polevl(y2, P0, 4) / _p1evl(y2, Q0, 8))
        x = x * ROOT_2PI
        return x

    x = math.sqrt(-2.0 * math.log(y))
    x0 = x - math.log(x) / x

    z = 1.0 / x
    if x < 8.0:                 # y > exp(-32) = 1.2664165549e-14
        x1 = z * _polevl(z, P1, 8) / _p1evl(z, Q1, 8)
    else:
        x1 = z * _polevl(z, P2, 8) / _p1evl(z, Q2, 8)

    x = x0 - x1
    if sign_flag != 0:
        x = -x

    return x

@njit
def erfinv(z):
    # Implementation from https://github.com/dougthor42/PyErf/blob/master/pyerf/pyerf.py
    if abs(z) > 1:
        raise ValueError("`z` must be between -1 and 1 inclusive")
    if z == 0:
        return 0
    if z == 1:
        return np.inf
    if z == -1:
        return -np.inf
    return ndtri((z + 1) / 2.0) / math.sqrt(2)

@njit
def ncdf(x, mean, std):
    return 0.5 * (1 + math.erf((x - mean) / np.sqrt(2) / std))

@njit
def ngmm_find_ppf(x, means, stds, weights, quantile):
    n = len(weights)
    assert len(means) == n
    assert len(stds) == n
    quantile_computed = 0
    for i in range(n):
        quantile_computed += weights[i] * ncdf(x, means[i], stds[i])
    return quantile - quantile_computed

@njit
def nppf(p, mean, std):
    return mean + std * np.sqrt(2) * erfinv(2 * p - 1)
