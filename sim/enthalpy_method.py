#-----------------------
# Enthalpy method module
#-----------------------

# Obsahuje definice funkci pro enthalpy metodu, formulace Cao, Equivalent heat capacity i nasi formulaci enthalpy metody

# Poznamky a dodelavky:
# 1. neni lepsi do formulace pro zhlazene fyzikalni parametry vkladat Expression misto Conditionalu?

import dolfin
import numpy as np

from ufl import tanh

# Global parameters
# Mollification constants
EPS = dolfin.Constant(0.4)
DEG = 'Cinf'
C_EPS=1.

#---------------------------------
# Definition of additional methods
#---------------------------------
# Auxiliary functions:
# Hyperbolic tangent:
# Note: this definition of hyperbolic tangent is unstable, use the one defined by ufl
#def tanh(x):
    #return (dolfin.exp(x) - dolfin.exp(-x))/(dolfin.exp(x) + dolfin.exp(-x))
# Sign function
def sign(x, x0=0.0):
    """ Give sign of the argument x-x0.

    Parameters
    ----------
    x: float 
            Input argument which position with respect to x0 we want to know
    x0: float, optional
            Center point.
    Returns
    -------
    -1,1,0: float
            A float representing the sign of the argument x-x0 
    """
    return dolfin.conditional(x<x0-dolfin.DOLFIN_EPS,-1,dolfin.conditional((x-x0)<dolfin.DOLFIN_EPS,0,1))

# Heaviside step function
def heaviside(x, x0=0.0, eps=EPS, deg=DEG):
    """Approximation of Heaviside function with center at x0 and half-width eps"""    

    # Discontinuous approximation
    def hs_disC():
        if type(x) == np.ndarray:
            y = 0.0*x
            for pos, val in enumerate(x):
                if abs(val - x0) < eps:
                    y[pos] = 0.5
                elif val > x0:
                    y[pos] = 1
                else:
                    y[pos] = 0
            return y
        return dolfin.conditional(abs(x-x0)<eps,0.5, dolfin.conditional(x>x0, 1, 0))
    
    # C0 approximation
    def hs_C0():
        if type(x) == np.ndarray:
            y = 0.0*x
            for pos, val in enumerate(x):
                if abs(val - x0) < eps:
                    y[pos] = (val-x0)/(2*eps)+0.5
                elif val > x0:
                    y[pos] = 1
                else:
                    y[pos] = 0
            return y
        return dolfin.conditional(abs(x-x0)<eps,(x-x0)/(2*eps)+0.5, dolfin.conditional(x>x0, 1, 0))

    # C1 approximation
    def hs_C1():
        if type(x) == np.ndarray:
            y = 0.0*x
            for pos, val in enumerate(x):
                if abs(val - x0) < eps:
                    y[pos] = -(val-x0)**3/(4*eps**3)+3*(val-x0)/(4*eps)+0.5
                elif val > x0:
                    y[pos] = 1
                else:
                    y[pos] = 0
            return y
        return dolfin.conditional(abs(x-x0)<eps, -(x-x0)**3/(4*eps**3)+3*(x-x0)/(4*eps)+0.5, dolfin.conditional(x > x0, 1, 0))

    # Cinf approximation
    def hs_Cinf():
        if type(x) == np.ndarray:
            return 0.5*np.tanh(2.5*(x-float(x0))/eps) + 0.5
        return 0.5*tanh(2.5*(x-x0)/eps) + 0.5
        
    degswitch = {
        'disC':hs_disC,
        'C0':hs_C0,
        'C1':hs_C1,
        'Cinf':hs_Cinf
        }
    return degswitch.get(deg,"Please enter 'CO','C1','Cinf', or 'exact'.")

# Dirac function
def df(x, x0=0.0, eps=EPS, deg=DEG):
    """Approximation of Dirac delta dist. with center at x0 and half-width eps"""
    # Discontinuous approximation
    def df_disC():
        if type(x) == np.ndarray:
            y = 0.0*x
            for pos, val in enumerate(x):
                if abs(val - x0) < eps:
                    y[pos] = 1.0/(2*eps)
                else:
                    y[pos] = 0
            return y
        return dolfin.conditional(abs(x-x0)<eps,1.0/(2*eps), 0.)
    
    # C0 approximation
    def df_C0():
        if type(x) == np.ndarray:
            y = 0.0*x
            for pos, val in enumerate(x):
                if abs(val - x0) < eps:
                    y[pos] = (1.0/(eps**2))*(eps-np.sign(val-x0)*(val-x0))
                else:
                    y[pos] = 0
            return y
        return dolfin.conditional(abs(x-x0)<eps,(1.0/(eps**2))*(eps-sign(x-x0)*(x-x0)), 0)

    # C1 approximation
    def df_C1():
        if type(x) == np.ndarray:
            y = 0.0*x
            for pos, val in enumerate(x):
                if abs(val - x0) < eps:
                    y[pos] = 15/(16*eps**5)*(((val-x0)+eps)**2*((val-x0)-eps)**2)
                else:
                    y[pos] = 0
            return y
        return dolfin.conditional(abs(x-x0)<eps, 15/(16*eps**5)*(((x-x0)+eps)**2*((x-x0)-eps)**2), 0)

    # Cinf approximation
    def df_Cinf():
        if type(x) == np.ndarray:
            return 1./(0.6*eps*np.sqrt(np.pi))*np.exp(-(x-x0)**2/(0.6**2*eps**2))
        return 1./(0.6*eps*np.sqrt(np.pi))*dolfin.exp(-(x-x0)**2/(0.6**2*eps**2))
        
    degswitch = {
        'disC': df_disC,
        'C0':df_C0,
        'C1':df_C1,
        'Cinf':df_Cinf
        }
    return degswitch.get(deg,"Please enter 'disC','CO','C1',or 'Cinf'.")

def mollify(xminus, xplus, x, x0=0.0, eps=EPS, deg=DEG):
    """Mollify the jump between xminus and xplus values."""
    return xminus*(1-heaviside(x, x0, eps, deg)()) + xplus*heaviside(x, x0, eps, deg)()

def dirac(xvalue, x, x0=0.0, eps=EPS, deg=DEG):
    """Return dirac with L1 norm of xvalue."""
    return xvalue*df(x,x0,eps,deg)()

def set_eps(hmax,theta_grad_max):
    # hmax=dolfin.MPI.max(mesh.mpi_comm(),mesh.hmax())
    # theta_norm=dolfin.project(dolfin.sqrt(dolfin.inner(dolfin.grad(theta),dolfin.grad(theta))),theta.function_space())
    # theta_grad_max=theta_norm.vector().norm('linf')
    global EPS
    EPS.assign(hmax*theta_grad_max/C_EPS)
