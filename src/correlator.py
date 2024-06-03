import argparse
import numpy
import scipy
import tqdm
import sympy as sym

from . import axion

def Re_integrand_linear(x0, x1, t0, t1, t2, t3, t4):
    '''
    Real component of $\tilde{G}$, using linear interpolation.
    params:
        x0, x1:
                x, y
                integration variables
        t0, t1:
                u, v
                shape parameters (k12/s, k34/s)
        t2, t3, t4:
                \\sigma, \\sigma*
                control points and values for interpolation of sigma, sigma*
    '''
    prefactor = numpy.exp(-x0 * t0) * numpy.exp(-x1 * t1) * (x0 ** -2) * (x1 ** -2)
    if x0 >= x1:
        s_p = numpy.interp(x0, t2, t3 * numpy.exp(-x0))
        s_n = numpy.interp(x1, t2, t4 * numpy.exp(-x1))
        return s_p * s_n * prefactor
    if x0 < x1:
        s_p = numpy.interp(x1, t2, t3 * numpy.exp(-x1))
        s_n = numpy.interp(x0, t2, t4 * numpy.exp(-x0))
        return s_p * s_n * prefactor

def Re_integrand_bspline(x0, x1, t0, t1, t2, t3, t4):
    '''
    Real component of $\\tilde{G}$, using B-spline interpolation.
    TODO [NS]: make this faster?
    params:
        x0, x1:
                x, y
                integration variables
        t0, t1:
                u, v
                shape parameters (k12/s, k34/s)
        t2, t3, t4:
                \\sigma, \\sigma*
                control points and values for interpolation of axion field
    '''
    prefactor = numpy.exp(-x0 * t0) * numpy.exp(-x1 * t1) * (x0 ** -2) * (x1 ** -2)
    if x0 >= x1:
        Re_s_p = scipy.interpolate.splev(x0, t2)
        Im_s_p = scipy.interpolate.splev(x0, t3)
        Re_s_n = scipy.interpolate.splev(x1, t2)
        Im_s_n = scipy.interpolate.splev(x1, t4)
        prop   = numpy.real((Re_s_p + 1j * Im_s_p) *
                            (Re_s_n + 1j * Im_s_n))
        return prop * prefactor
    if x0 < x1:
        Re_s_p = scipy.interpolate.splev(x1, t2)
        Im_s_p = scipy.interpolate.splev(x1, t3)
        Re_s_n = scipy.interpolate.splev(x0, t2)
        Im_s_n = scipy.interpolate.splev(x0, t4)
        prop   = numpy.real((Re_s_p + 1j * Im_s_p) *
                            (Re_s_n + 1j * Im_s_n))
        return prop * prefactor

def Re_propagator_simpson(s, s_):
    '''
    Construct the propagator for a fast Simpson-like integration.

    This function is separate as it does not depend on u, v.
    params:
        s, s_:
            \\sigma, \\sigma*
    '''
    # The outer product of s, s_ gives 2-d grid of \sigma(x) * \sigma*(y).
    # The numpy.triu, numpy.tril resp. zero out the components of the
    #   matrix where x >= y, vice versa.
    # The numpy.diag of the original matrix is subtracted to cancel the
    #   double counting when x == y.
    # This is a fast (vectorised) construction of \tilde{G}.
    P   = (numpy.triu(numpy.outer(s, s_)) +
           numpy.tril(numpy.outer(s_, s)) -
           numpy.diag(s_ * s))
    return P

def Re_integrand_simpson(t, P, u, v):
    '''
    Construct the integrand for a fast Simpson-like integration.

    params:
        t:
            control points for \\sigma, \\sigma*
        P:
            propagator matrix
        u, v:
            shape parameters

    '''
    N = len(t)
    xv   = t ** -2 * numpy.exp(-u * t)
    yv   = t ** -2 * numpy.exp(-v * t)
    prefactors =  numpy.outer(xv, yv)
    G = prefactors * P
    return G

class Correlator:
    def __init__(self, config):
        self.solver_opts = argparse.Namespace(**config['solver_opts']['correlator'])
        self.parameters  = argparse.Namespace(**config['parameters'])
        self.axion       = axion.Axion(config)

    def __repr__(self):
        return repr(self.axion)

    def _Re_integrate_single(self, u):
        '''
        Run the integration over a series of u values determined by configuration file
        '''
        interp = self.solver_opts.interp

        v  = self.parameters.v

        t  = self.axion.t
        s_re  = self.axion.s_re
        s_im  = self.axion.s_im

        opts = {
            'limit' : 2 * len(t),
            'points': t,
            'epsrel': self.solver_opts.epsrel,
            'epsabs': self.solver_opts.epsabs,
        }
        x_min = 10 ** self.solver_opts.log_x_min
        x_max = 10 ** self.solver_opts.log_x_max

        if interp == 'linear':
            args = (u, v, t, s_re + s_im, s_re - s_im)
            F, err = scipy.integrate.nquad(Re_integrand_linear, [[x_min, x_max], [x_min, x_max]], opts=opts, args=args)
        elif interp == 'spline':
            args = (u, v, t, s_re + s_im, s_re - s_im)
            F, err = scipy.integrate.nquad(Re_integrand_bspline, [[x_min, x_max], [x_min, x_max]], opts=opts, args=args)
        elif interp == 'simpson':
            P    = Re_propagator_simpson(s_re + s_im, s_re - s_im)
            G    = Re_integrand_simpson(t, P, u, v)
            quad = scipy.integrate.simps
            F, err = quad([quad(G_x, t) for G_x in G], t), None
        else:
            raise NotImplementedError("Integration interp mode should be one of (linear, spline, simpson)")
        return F

    def integrate(self):
        '''
        Run the integration over a series of u values determined by configuration file
        '''
        log_u_min = self.solver_opts.log_u_min
        log_u_max = self.solver_opts.log_u_max
        n_pts = self.solver_opts.u_n_pts
        arr_u = numpy.logspace(log_u_min, log_u_max, n_pts, base=10)
        u_min = 10 ** self.solver_opts.log_u_min
        u_max = 10 ** self.solver_opts.log_u_max
        arr_F = numpy.zeros_like(arr_u)
        for u_ix, u in enumerate(tqdm.tqdm(arr_u)):
            F = self._Re_integrate_single(u)
            arr_F[u_ix] = F
        return arr_u, arr_F


