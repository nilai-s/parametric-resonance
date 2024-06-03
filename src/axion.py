import numpy
import scipy
import sympy as sym
import argparse

'''
Our equation is written as follows in Euclidean time
    0 = - s''
        + {2 / x} * s'
        + {1 - (a_1 / x ^ 2)} * s
        - {(g * g * a_1 / x ^ 2) * cos^2(a_2 * log(ix / a_3))} * s

Substituting f(x) = e^x * s(x):
    0 = - (f'' - 2f' + f)
        + {2 / x} * (f' - f)
        + {1 - (a_1 / x ^ 2)} * f
        - {(g * g * a_1 / x ^ 2) * cos^2(a_2 * log(ix / a_3))} * f
'''

class Axion:
    def __init__(self, config):
        self.solver_opts = argparse.Namespace(**config['solver_opts']['axion'])
        self.parameters  = argparse.Namespace(**config['parameters'])
        # redefine some parameters for convenience
        self.a_1 = (self.parameters.m0 / self.parameters.H) ** 2
        self.a_2 = (self.parameters.w  / self.parameters.H)
        self.a_3 = (self.parameters.a)
        self.g   = (self.parameters.g)

        if self.solver_opts.solver == 'numeric':
            self.solve_ode_numeric()
        elif self.solver_opts.solver == 'algebraic':
            self.solve_ode_algebraic()
        else:
            raise NotImplementedError('ODE solver must be (numeric, algebraic)')

    def __repr__(self):
        return 'g=%.2f,mu=%.2f' % (self.g, numpy.sqrt(self.a_1))

    def solve_ode_algebraic(self):
        '''
        Solve the ODE algebraically with sympy.
        Note that the oscillatory component of the potential must be zeroed.
        '''
        assert self.g == 0
        # variables
        x = sym.symbols('x', complex=True)
        s = sym.Function('s', complex=True)(x)

        # parameters
        a_3 = self.a_3
        a_2 = self.a_2
        a_1 = self.a_1
        g   = 0

        # derivatives
        spp = sym.Derivative(s, x, x)
        sp  = sym.Derivative(s, x)

        # terms in the equation of motion
        term_1  = -spp * (1)
        term_2  =  sp  * (2 / x)
        term_3  =  s   * (1 - a_1 / (x ** 2))
        term_4  = -s   * (g * g * a_1 / (x ** 2) *
                          sym.cos( a_2 * sym.log(sym.I * x / a_3 ) ) *
                          sym.cos( a_2 * sym.log(sym.I * x / a_3 ) ))
        eq  = term_1 + term_2 + term_3 + term_4

        # initial conditions
        x_0  = 10
        ics  = {
            s.subs({x: x_0})  : (x_0) * sym.exp(-x_0),
            sp.subs({x: x_0}) : (1 - x_0) * sym.exp(-x_0),
        }

        # return inputs to dsolve
        eq   = sym.Eq(eq, 0)
        func = s
        ics  = ics
        solution = sym.solvers.ode.dsolve(eq, func=func, ics=ics)

        # extract values
        log_x_min = self.solver_opts.log_x_min
        log_x_max = self.solver_opts.log_x_max
        n_pts = self.solver_opts.x_n_pts
        xx = numpy.logspace(log_x_min, log_x_max, n_pts, base=10)
        x_min = 10 ** self.solver_opts.log_x_min
        x_max = 10 ** self.solver_opts.log_x_max
        self.t    = xx
        func = lambda x_: complex(solution.rhs.subs({x: x_}).evalf())
        self.s    = numpy.array([func(x_) for x_ in xx])
        self.s_re = numpy.real(self.s)
        self.s_im = numpy.imag(self.s)
        return solution

    def solve_ode_numeric(self):
        '''
        Solve the ODE numerically with scipy.
        '''
        # parameters
        a_3  = self.a_3
        a_2  = self.a_2
        a_1  = self.a_1
        g    = self.g

        def d(x, z):
            '''
            Compute the derivative of vector z in terms of x.

            z is defined as the linear system:
                z  := [ f  , f'  ]
                z' := [ f' , f'' ]
            Note that:
                z'[0] = f'  = z[1]
                z'[1] = f'' = ...
            This is necessary for scipy numerical solver to work.
            '''
            fp  =  z[1]
            fpp = (
                + 2 * z[1] - z[0]
                + (z[1] - z[0]) * (2 / x)
                +  z[0] * (1 - a_1 / (x ** 2))
                -  z[0] * (g * g * a_1 / (x ** 2)) * (numpy.cos( a_2 * numpy.log(1j * x / a_3) ) ** 2)
            )
            return numpy.array((fp, fpp))

        # initial conditions
        log_x_min = self.solver_opts.log_x_min
        log_x_max = self.solver_opts.log_x_max
        n_pts = self.solver_opts.x_n_pts
        xx = numpy.logspace(log_x_min, log_x_max, n_pts, base=10)
        xx = xx[::-1]
        x_min = 10 ** self.solver_opts.log_x_min
        x_max = 10 ** self.solver_opts.log_x_max

        f_ic  = complex( x_max )
        fp_ic = complex( 1.0 )
        z_ic  = numpy.array((f_ic, fp_ic))

        # numerically integrate
        sol = scipy.integrate.solve_ivp(d, [x_max, x_min], z_ic, t_eval=xx, atol=1e-8)

        # flip the array since we integrated from early to late times
        # but we want to plot numerically lower x-values first
        self.t = sol.t[::-1]
        self.s_re = numpy.real(sol.y[0, ::-1])
        self.s_im = numpy.imag(sol.y[0, ::-1])
        return sol




