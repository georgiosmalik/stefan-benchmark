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
EPS = 0.4
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

def set_eps(mesh,theta):
    h_min=mesh.hmin()
    theta_norm=dolfin.project(dolfin.sqrt(dolfin.inner(dolfin.grad(theta),dolfin.grad(theta))),theta.function_space())
    theta_grad_max=theta_norm.vector().norm('linf')
    global EPS
    EPS=C_EPS*h_min*theta_grad_max
# #---------------------------------
# # Enthalpy method formulation from Cao, 1990:
# # Source term:
# def s(theta, theta0 = theta_m, eps = eps):
#     return conditional(abs(theta-theta0)<eps,c_m*eps + L_m/2, conditional(theta > theta0, c_s*eps + L_m, c_s*eps))
# def enth(theta, theta0 = theta_m, eps = eps):
#     return c(theta, theta0, eps, type="cao")*(theta-theta0) + s(theta, theta0)
# #---------------------------------
# # Mollified material characteristics:
# # Kowalewski experiment state equations for water:
# def rho_l_kowal(x):
#     return 999.840281167108 + 0.0673268037314653*(x - 273.15) - 0.00894484552601798*(x-273.15)**2 + 8.78462866500416e-5*(x-273.15)**3 - 6.62139792627547e-7*(x-273.15)**4

# def alpha_l_kowal(x):
#     return -(0.0673268037314653 - 2*0.00894484552601798*(x-273.15) + 3*8.78462866500416e-5*(x-273.15)**2 - 4*6.62139792627547e-7*(x-273.15)**3)/rho_l_kowal(x)

# def c_l_kowal(x):
#     return 8958.66-40.534*x + 1.1234e-1*x**2-1.01379e-4*x**3

# def k_l_kowal(x):
#     return 0.566*(1 + 0.001*x)

# def mu_l_kowal(x):
#     return 1.79e-3*exp(6.18e7*(1/(x**3)-1/(273.15**3)))
# #---------------------------------
# # Mollified density:
# def rho(x, x0 = theta_m, eps = eps, deg = deg, type=None):
#     if type=="kowal":
#         return rho_l_kowal(x)*hs(x, x0, eps, deg) + rho_s*(Constant(1)-hs(x, x0, eps, deg))
#     elif type=="dana":
#         return rho_w(x)*hs(x, x0, eps, deg) + rho_s*(Constant(1)-hs(x, x0, eps, deg))
#     return rho_l*hs(x, x0, eps, deg) + rho_s*(Constant(1.)-hs(x, x0, eps, deg))

# # Mollified heat capacity:
# def c(theta, theta0 = theta_m, eps = eps, deg = deg, type=None):
#     if type=="kowal":
#         return c_l_kowal(theta)*hs(theta, theta0, eps, deg) + L_m*df(theta, theta0, eps, deg) + c_s*(Constant(1)-hs(theta, theta0, eps, deg))
#     if type=="cao":
#         return conditional(abs(theta-theta0)<eps,c_m + L_m/(2*eps), conditional(theta > theta0, c_l, c_s))
#     return c_l*hs(theta, theta0, eps, deg) + L_m*df(theta, theta0, eps, deg) + c_s*(Constant(1)-hs(theta, theta0, eps, deg))

# # Mollified product of capacity and density:
# def rhoc(x, x0 = theta_m, eps = eps, deg = deg, type=None):
#     if type=="kowal":
#         return rho_l*c_l_kowal(x)*hs(x, x0, eps, deg) + L_m*(rho_l+rho_s)/2*df(x, x0, eps, deg) + rho_s*c_s*(Constant(1)-hs(x, x0, eps, deg))
#     elif type=="dana":
#         return rho_w(x)*c_l_kowal(x)*hs(x, x0, eps, deg) + L_m*rho_l*df(x, x0, eps, deg) + rho_s*c_s*(Constant(1)-hs(x, x0, eps, deg))
#     return rho_l*c_l*hs(x, x0, eps, deg) + L_m*rho_l*df(x, x0, eps, deg) + rho_s*c_s*(Constant(1)-hs(x, x0, eps, deg))

# # Model of temperature dependent density of water for Kowalski benchmark (see Danaila in refs):
# def rho_w(theta):
#     # Density function parameters:
#     theta_max = Constant(277.1793)      # Temperature of water with max density
#     rho_max = Constant(999.972)         # Reference density at theta_max
#     w_coeff = 9.2793e-6                 # Multiplicative coefficient [K^(-q)]
#     q_coeff = 1.894816                  # Exponent of temperature difference
#     return rho_max*(1 - w_coeff*(abs(theta - theta_max))**q_coeff)

# # Effective value of density for the buoyancy term
# def rho_w_eff(x, x0 = theta_m, eps = eps, deg = deg):
#     return rho_w(x)*hs(x, x0, eps, deg) + rho_s*(Constant(1.)-hs(x, x0, eps, deg))

# # Mollified heat conductivity:
# def k(x, x0 = theta_m, eps = eps, deg = deg, type=None):
#     if type == "kowal":
#         return k_l_kowal(x)*hs(x, x0, eps, deg) + k_s*(Constant(1)-hs(x, x0, eps, deg))
#     return k_l*hs(x, x0, eps, deg) + k_s*(Constant(1)-hs(x, x0, eps, deg))

# # Mollified viscosity:
# def mu(x, x0 = theta_m, eps = eps, deg = deg, type = None):
#     if type == "kowal":
#        return mu_l_kowal(x)*hs(x, x0, eps, deg) + mu_s*(Constant(1.)-hs(x, x0, eps, deg))
#     return mu_l*hs(x, x0, eps, deg) + mu_s*(Constant(1.)-hs(x, x0, eps, deg))

# # Mollified logarithmic viscosity:
# def mu_log(x, x0 = theta_m, eps = eps, deg = deg):
#        return exp(np.log(float(mu_l))*hs(x, x0, eps, deg) + math.log(float(mu_s))*(Constant(1.)-hs(x, x0, eps, deg)))

# # Mollified expansion coefficient:
# def alpha(x, x0 = theta_m, eps = eps, deg = deg, type = None):
#     if type == "kowal":
#         return alpha_l_kowal(x)*hs(x, x0, eps, deg) + alpha_s*(Constant(1.)-hs(x, x0, eps, deg))
#     return alpha_l*hs(x, x0, eps, deg) + alpha_s*(Constant(1.)-hs(x, x0, eps, deg))
# #---------------------------------
# # Time step size
# def cfl_dt(uv,mesh,*timescales):
#        """Computes the time step using CFL condition"""
#        C_CFL_v = 0.5
#        C_CFL_T = 1e-1
#        #h_min = MPI.min(mesh.mpi_comm(),mesh.hmin())
#        h_min = mesh.hmin()
#        #v_max = MPI.max(mesh.mpi_comm(),numpy.abs(uv.vector().array()).max())
#        v_max = norm(uv.vector(),'linf')
#        ts = [C_CFL_v*h_min/v_max]
#        for tau in timescales:
#            ts.append(C_CFL_T*tau)
#        # value of CFL
#        print(ts)
#        dt = min(ts)
#        return dt
# #=================================
