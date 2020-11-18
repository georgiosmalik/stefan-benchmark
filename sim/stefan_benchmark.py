# ----------------------------------
# Stefan benchmark simulation module
# ----------------------------------

import dolfin
import numpy as np
import csv

import pre.stefan_mesh as smsh
import sim.params as prm
import sim.enthalpy_method as em
import post.my_plot as mplt
import post.stefan_plot as splt

from scipy.special import erf, erfc, expi, gamma, gammaincc
from scipy.optimize import fsolve
from math import floor, ceil

import time

dolfin.parameters['form_compiler']['quadrature_degree']=4

dolfin.set_log_level(30)

#------------------------------------
# Global parameters of the simulation
#------------------------------------

# Dimension of problem formulation
DIM=0

# Type of boundary formulation (e.g. "DN"=Dirichlet-Neumann)
BOUNDARY_FORMULATION="NN"

# Two basic types of implementations
METHODS=['EHC','TTM']

# Degree of finite element spaces:
DEGREE=1

# Starting and ending radius of simulation
R_START=0.2
R_END=0.8

# Nonlinear solver parameters
NEWTON_PARAMS=dolfin.Parameters("newton_solver")
NEWTON_PARAMS.add("linear_solver","bicgstab")
NEWTON_PARAMS.add("absolute_tolerance",1e-5)
NEWTON_PARAMS.add("maximum_iterations",25)

# Specify data output
SAVE_DAT = False
TEMP_TXT_DAT = True
SAVE_FRONT_POS_TXT = True

# Logging for cluster computing
LOG = False

# Types of simulation
CONVERGENCE = False
STABILITY = False

# Temporal discretization scheme for EHC model (THETA = 0.5 is Crank-Nicholson, THETA=1 is fully implicit)
THETA=0.5
# ===================================

# ------------------------------------------------------
# Analytic solution of radially symmetric Stefan problem
# ------------------------------------------------------

def stefan_analytic_sol(dim, ploteq=False):
    """Return analytic solution of the radially symmetric Stefan problem."""

    def graph_transcendental_eq(lambda_,fun,dim):
        x_arr = np.linspace(0,2*lambda_,101)
        y_arr = fun(x_arr[2:])
        mplt.graph1d([[x_arr[2:],y_arr]],
                     axlabels = [r"$\lambda$",r"$F_"+str(dim)+"(\lambda)$"],
                     savefig = {"width":"IJHMT","name":"./out/fig/"+str(dim)+"d/transcendetal_eq_plot_"+str(dim)+"d.pdf"},
                     title = "Plot of transcendental equation for $d = "+str(dim)+"$"
        )
        
    def theta_sol_1d():
                
        def transcendental_eq_1d(savefig):

            f=lambda x: prm.rho*prm.L_m*x-prm.q_0*np.exp(-x*x/prm.kappa_l)+prm.k_s/np.sqrt(prm.kappa_s*np.pi)*(prm.theta_m - prm.theta_i)/erfc(x/np.sqrt(prm.kappa_s))*np.exp(-x*x/prm.kappa_s)
            lambda_ = fsolve(f,0.00001,xtol=1e-10)
            if savefig:
                ax, _ = graph_transcendental_eq(lambda_,f,1)
                ax.axhline(y=0, lw=1, color='k')
            return lambda_[0]

        lambda_=transcendental_eq_1d(ploteq)

        # Analytic solution cpp code:
        code_analytic="""x[0] < 2*lambda_*sqrt(t) ? q_0*sqrt(kappa_l*pi)/k_l*(erf(lambda_/sqrt(kappa_l))-erf(x[0]/(2*sqrt(kappa_l*t))))+theta_m : theta_i + (theta_m - theta_i)/erfc(lambda_/sqrt(kappa_s))*erfc(x[0]/(2*sqrt(t*kappa_s)))"""
        
        theta_analytic = dolfin.Expression(code_analytic,lambda_=lambda_, t=0.1, q_0=prm.q_0, theta_m=prm.theta_m, theta_0=prm.theta_0, theta_i=prm.theta_i, k_l=prm.k_l, kappa_l=prm.kappa_l, kappa_s=prm.kappa_s, degree=3)

        # cpp code heat flux:
        code_flux='-k*C1*exp(-r*r/(4*t*kappa))/sqrt(t)'

        # Heat influx:
        q_in=dolfin.Expression(code_flux, t=0.1, k=prm.k_l, C1=-prm.q_0/prm.k_l, r=prm.R1, kappa=prm.kappa_l, degree=0)
        
        q_in_k=dolfin.Expression(code_flux, t=0.1, k=prm.k_l, C1=-prm.q_0/prm.k_l, r=prm.R1, kappa=prm.kappa_l, degree=0)

        # Heat outflux:
        q_out=dolfin.Expression(code_flux, t=0.1, k=prm.k_s, C1=(prm.theta_m-prm.theta_i)/(erfc(lambda_/np.sqrt(prm.kappa_s))*np.sqrt(np.pi*prm.kappa_s)), r=prm.R2, kappa=prm.kappa_s, degree=0)
        
        q_out_k=dolfin.Expression(code_flux, t=0.1, k=prm.k_s, C1=(prm.theta_m-prm.theta_i)/(erfc(lambda_/np.sqrt(prm.kappa_s))*np.sqrt(np.pi*prm.kappa_s)), r=prm.R2, kappa=prm.kappa_s, degree=0)
        
        return lambda_, theta_analytic, q_in, q_in_k, q_out, q_out_k

    def theta_sol_2d():

        def transcendental_eq_2d(savefig):
            
            f = lambda x : prm.rho*prm.L_m*x**2 - prm.q_0/(4*np.pi)*np.exp(-(x**2)/(prm.kappa_l)) - prm.k_s*(prm.theta_m-prm.theta_i)*np.exp(-(x**2)/prm.kappa_s)/(expi(-(x**2)/prm.kappa_s))
            
            lambda_ = fsolve(f,0.00001,xtol=1e-10)
            
            if savefig:
                ax, _ = graph_transcendental_eq(lambda_,f,2)
                ax.axhline(y=0, lw=1, color='k')
                
            return lambda_[0]

        lambda_=transcendental_eq_2d(ploteq)

        # Analytic solution cpp code (pybind11):
        code_analytic='''
        #include <pybind11/pybind11.h>
        #include <pybind11/eigen.h>
        namespace py = pybind11;

        #include <dolfin/function/Expression.h>
        #include <dolfin/mesh/MeshFunction.h>
        #include <math.h>
        #include <boost/math/special_functions/expint.hpp>
        using boost::math::expint;

        class StefanAnalytic2d : public dolfin::Expression
        {
        public:

          double t, theta_i, theta_m, kappa_l, kappa_s, lambda_, c_2d;
          // Analytical solution, returns one value
          StefanAnalytic2d() : dolfin::Expression() {};
          // Function for evaluating expression
          void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
          {
            double f_l = (x[0]*x[0]+x[1]*x[1])/(4*kappa_l*t) ;
            double f_s = (x[0]*x[0]+x[1]*x[1])/(4*kappa_s*t) ;
            if ( sqrt(x[0]*x[0]+x[1]*x[1]) <= 2*lambda_*sqrt(t) ) {
               values[0] = theta_m + c_2d*(expint(-lambda_*lambda_/kappa_l) - expint(-f_l));
            }
            else {
               values[0] = theta_i - (theta_i - theta_m)/expint(-lambda_*lambda_/kappa_s)*expint(-f_s);
            }
          }
        };

        PYBIND11_MODULE(SIGNATURE, m)
        {
          py::class_<StefanAnalytic2d, std::shared_ptr<StefanAnalytic2d>, dolfin::Expression>
            (m, "StefanAnalytic2d")
            .def(py::init<>())
            .def_readwrite("kappa_l", &StefanAnalytic2d::kappa_l)
            .def_readwrite("kappa_s", &StefanAnalytic2d::kappa_s)
            .def_readwrite("lambda_", &StefanAnalytic2d::lambda_)
            .def_readwrite("theta_m", &StefanAnalytic2d::theta_m)
            .def_readwrite("c_2d", &StefanAnalytic2d::c_2d)
            .def_readwrite("theta_i", &StefanAnalytic2d::theta_i)
            .def_readwrite("t", &StefanAnalytic2d::t);
        }
        '''
        
        theta_analytic = dolfin.CompiledExpression(dolfin.compile_cpp_code(code_analytic).StefanAnalytic2d(),
                                      kappa_l=prm.kappa_l,
                                      kappa_s=prm.kappa_s,
                                      theta_m=prm.theta_m,
                                      c_2d=prm.q_0/(4*np.pi*prm.k_l),
                                      theta_i=prm.theta_i,
                                      lambda_=lambda_,
                                      t=0.1,
                                      degree=3)

        # 2d heat flux cpp code:
        code_flux='-k*c_2d*exp(-r*r/(4*kappa*t))/r'

        # Heat influx:
        q_in=dolfin.Expression(code_flux, t=0.1, k=prm.k_l, c_2d=-prm.q_0/(2*np.pi*prm.k_l), r=prm.R1, kappa=prm.kappa_l, degree=0)

        q_in_k=dolfin.Expression(code_flux, t=0.1, k=prm.k_l, c_2d=-prm.q_0/(2*np.pi*prm.k_l), r=prm.R1, kappa=prm.kappa_l, degree=0)

        # Heat outflux:
        q_out=dolfin.Expression(code_flux, t=0.1, k=prm.k_s, c_2d=(-2)*(prm.theta_m-prm.theta_i)/expi(-lambda_**2/prm.kappa_s), r=prm.R2, kappa=prm.kappa_s, degree=0)

        q_out_k=dolfin.Expression(code_flux, t=0.1, k=prm.k_s, c_2d=(-2)*(prm.theta_m-prm.theta_i)/expi(-lambda_**2/prm.kappa_s), r=prm.R2, kappa=prm.kappa_s, degree=0)
        
        return lambda_, theta_analytic, q_in, q_in_k, q_out, q_out_k

    def theta_sol_3d():

        def transcendental_eq_3d(savefig):
            
            f = lambda x : prm.rho*prm.L_m*x**3 - prm.q_0/(16*np.pi)*np.exp(-(x**2)/prm.kappa_l) + prm.k_s*np.sqrt(prm.kappa_s)*(prm.theta_m - prm.theta_i)*np.exp(-(x**2)/prm.kappa_s)/((-2)*gamma(0.5)*gammaincc(0.5,x**2/prm.kappa_s) + 2*np.sqrt(prm.kappa_s)/x*np.exp(-x**2/prm.kappa_s))

            lambda_ = fsolve(f,0.00001,xtol=1e-10)
            
            if savefig:
                ax, _ = graph_transcendental_eq(lambda_,f,3)
                ax.axhline(y=0, lw=1, color='k')
            return lambda_[0]

        lambda_=transcendental_eq_3d(ploteq)

        # Analytic solution cpp code (pybind11):
        code_analytic='''
        #include <pybind11/pybind11.h>
        #include <pybind11/eigen.h>
        namespace py = pybind11;

        #include <dolfin/function/Expression.h>
        #include <dolfin/mesh/MeshFunction.h>
        #include <math.h>
        #include <boost/math/special_functions/gamma.hpp>
        using boost::math::tgamma;

        class StefanAnalytic3d : public dolfin::Expression
        {
        public:

          double t, theta_i, theta_m, kappa_l, kappa_s, lambda_, c_3d;
          
          // Analytical solution, returns one value
          StefanAnalytic3d() : dolfin::Expression() {};
          // Function for evaluating expression
          void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
          {
            double f_l = (x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/(4*kappa_l*t) ;
            double f_s = (x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/(4*kappa_s*t) ;
            if ( sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]) <= 2*lambda_*sqrt(t) ) {
               values[0] = theta_m + c_3d*((-2*tgamma(0.5,f_l) + 2*sqrt(1/f_l)*exp(-f_l))  - (-2*tgamma(0.5,lambda_*lambda_/kappa_l) + 2*sqrt(kappa_l)/lambda_*exp(-lambda_*lambda_/kappa_l)));
            }
            else {
               values[0] = theta_i - (theta_i - theta_m)/(-2*tgamma(0.5,lambda_*lambda_/kappa_s) + 2*sqrt(kappa_s)/lambda_*exp(-lambda_*lambda_/kappa_s))*(-2*tgamma(0.5,f_s) + 2*sqrt(1/f_s)*exp(-f_s));
            }
          }
        };

        PYBIND11_MODULE(SIGNATURE, m)
        {
          py::class_<StefanAnalytic3d, std::shared_ptr<StefanAnalytic3d>, dolfin::Expression>
            (m, "StefanAnalytic3d")
            .def(py::init<>())
            .def_readwrite("kappa_l", &StefanAnalytic3d::kappa_l)
            .def_readwrite("kappa_s", &StefanAnalytic3d::kappa_s)
            .def_readwrite("lambda_", &StefanAnalytic3d::lambda_)
            .def_readwrite("theta_m", &StefanAnalytic3d::theta_m)
            .def_readwrite("c_3d", &StefanAnalytic3d::c_3d)
            .def_readwrite("theta_i", &StefanAnalytic3d::theta_i)
            .def_readwrite("t", &StefanAnalytic3d::t);
        }
        '''

        # Compile cpp code for dolfin:
        theta_analytic = dolfin.CompiledExpression(dolfin.compile_cpp_code(code_analytic).StefanAnalytic3d(),
                                                   kappa_l=prm.kappa_l,
                                                   kappa_s=prm.kappa_s,
                                                   theta_m=prm.theta_m,
                                                   c_3d=prm.q_0/(16*np.pi*np.sqrt(prm.kappa_l)*prm.k_l),
                                                   theta_i=prm.theta_i,
                                                   lambda_=lambda_,
                                                   t=0.1,
                                                   degree=3)

        # 3d heat flux cpp code:
        code_flux='-k*c_3d*exp(-r*r/(4*kappa*t))*sqrt(4*kappa*t)/(r*r)'

        # Heat influx:
        q_in=dolfin.Expression(code_flux, t=0.1, k=prm.k_l, c_3d=-prm.q_0/(8*np.pi*prm.k_l*np.sqrt(prm.kappa_l)), r=prm.R1, kappa=prm.kappa_l, degree=0)

        q_in_k=dolfin.Expression(code_flux, t=0.1, k=prm.k_l, c_3d=-prm.q_0/(8*np.pi*prm.k_l*np.sqrt(prm.kappa_l)), r=prm.R1, kappa=prm.kappa_l, degree=0)

        # Heat outflux:
        q_out=dolfin.Expression(code_flux, t=0.1, k=prm.k_s, c_3d=-(prm.theta_i-prm.theta_m)/((-1)*gamma(0.5)*gammaincc(0.5,lambda_**2/prm.kappa_s) + np.sqrt(prm.kappa_s)/lambda_*np.exp(-lambda_**2/prm.kappa_s)), r=prm.R2, kappa=prm.kappa_s, degree=0)
        
        q_out_k=dolfin.Expression(code_flux, t=0.1, k=prm.k_s, c_3d=-(prm.theta_i-prm.theta_m)/((-1)*gamma(0.5)*gammaincc(0.5,lambda_**2/prm.kappa_s) + np.sqrt(prm.kappa_s)/lambda_*np.exp(-lambda_**2/prm.kappa_s)), r=prm.R2, kappa=prm.kappa_s, degree=0)
        
        return lambda_, theta_analytic, q_in, q_in_k, q_out, q_out_k
    
    dimswitch = {
        1:theta_sol_1d,
        2:theta_sol_2d,
        3:theta_sol_3d
        }
    return dimswitch.get(dim, "Please enter 1d, 2d, or 3d.")
# ======================================================

# ------------------------------------------------
# Finite element implementation of enthalpy method
# ------------------------------------------------

def stefan_benchmark_sim(mesh, boundary, n, dx, ds, lambda_, theta_analytic, q_in, q_in_k, q_out, q_out_k , methods):

    # MPI objects:
    comm=dolfin.MPI.comm_world
    bbox=mesh.bounding_box_tree()

    global rank
    rank=dolfin.MPI.rank(mesh.mpi_comm())

    # Mesh parameters:
    # by dolfin:
    hmin = dolfin.MPI.min(mesh.mpi_comm(),mesh.hmin())
    
    hmax = dolfin.MPI.max(mesh.mpi_comm(),mesh.hmax())

    # custom:
    maxlength = 0.
    minlength = prm.R2
    avlength = 0
    num_edges = 0
    for f in dolfin.facets(mesh):
        for e in dolfin.edges(f):
            if (e.length()<minlength): minlength = e.length()
            if (e.length()>maxlength): maxlength = e.length()
            avlength = avlength + e.length()
            num_edges = num_edges + 1
    avlength = dolfin.MPI.sum(mesh.mpi_comm(),avlength)
    num_edges = dolfin.MPI.sum(mesh.mpi_comm(),num_edges)
    avlength = avlength/num_edges
    
    def stefan_loop_timesets():

        # Set start and end time of simulation
        global t_0
        t_0=(R_START/(2*lambda_))**2
        
        t_max=(R_END/(2*lambda_))**2

        # Set timestep based on standart CFL condition:
        if not ('dt' in globals()):
            
            # Maximal velocity of melting front:
            vmax=lambda_/np.sqrt(t_0)
            
            global dt
            dt = em.get_delta_t_cfl(hmin, vmax)

        # Set timeset for simulation
        sim_timeset=np.arange(t_0,t_max,dt)[1:]

        # Set timeset for data output
        numdats=100

        if numdats >= len(sim_timeset):
            dat_timeset=sim_timeset
        else:
            idx_dats=np.round(np.linspace(0,len(sim_timeset)-1,numdats)).astype(int)
            dat_timeset=sim_timeset[idx_dats]

        # Set timeset for plotting
        numplots=3
        
        idx_plots=np.round(np.linspace(0,len(dat_timeset)-1,numplots)).astype(int)
        
        plot_timeset=dat_timeset[idx_plots]
        
        return sim_timeset, dat_timeset, plot_timeset

    def stefan_function_spaces(degree=DEGREE):

        T_ele = dolfin.FiniteElement("CG", mesh.ufl_cell(), degree)
        T = dolfin.FunctionSpace(mesh, T_ele)
        
        boundary_conditions=[dolfin.DirichletBC(T,theta_analytic,boundary[0],1),dolfin.DirichletBC(T,theta_analytic,boundary[0],2)]
        
        theta = dolfin.Function(T)
        _theta = dolfin.TrialFunction(T)
        theta_ = dolfin.TestFunction(T)
            
        return (T,boundary_conditions,theta,_theta,theta_)

    def stefan_form_update(t):
        theta_analytic.t=t
        q_in.t=t
        q_out.t=t

    def stefan_form_update_previous(t):
        q_in_k.t=t
        q_out_k.t=t

    def stefan_front_position(theta):
        vol_ice=dolfin.assemble(em.mollify(1,0,theta-prm.theta_m,x0=0,eps=em.EPS,deg='C0')*dx)
        
        def front_pos_1d():
            return prm.R2-vol_ice
        def front_pos_2d():
            return np.sqrt(prm.R2**2-vol_ice/np.pi)
        def front_pos_3d():
            # Mesh for 3d is only one eight of entire ball
            return np.cbrt(prm.R2**3-6*vol_ice/np.pi)

        switch = {
            1:front_pos_1d,
            2:front_pos_2d,
            3:front_pos_3d
        }
        
        return switch.get(DIM)
    
    def stefan_loop():
        
        def stefan_problem_form(method):

            def stefan_boundary_values(theta_test,bc):
                
                formulation={"DN":0,"ND":1,"NN":2,"DD":0.5}
                i=formulation[BOUNDARY_FORMULATION]
                
                q_form = [q_out*theta_test*ds(2),q_in*theta_test*ds(1)][floor(-1.5+i):ceil(0.5+i)]

                q_form_k = [q_out_k*theta_test*ds(2),q_in_k*theta_test*ds(1)][floor(-1.5+i):ceil(0.5+i)]
                
                bc_form=bc[floor(0+i):ceil(1+i)]

                return q_form, q_form_k, bc_form

            # Define mollified parameters:
            def k_eff(theta,deg=em.DEG):
                return em.mollify(prm.k_s,prm.k_l,theta,x0=prm.theta_m,deg=deg)

            def c_p_eff(theta,deg=em.DEG):
                    return em.mollify(prm.cp_s,prm.cp_l,theta,x0=prm.theta_m,deg=deg)+em.dirac(prm.L_m,theta,x0=prm.theta_m,deg=deg)

            def stefan_form_ehc():

                # Define functions and spaces:
                (T,bcs,theta,_theta,theta_)=stefan_function_spaces()
                
                # Set initial condition:
                theta_k=dolfin.project(theta_analytic,T,solver_type="cg",preconditioner_type="hypre_amg")

                # Set boundary terms
                q_form, q_form_k, bc_form=stefan_boundary_values(theta_,bcs)

                # Partial THETA time discretization scheme:
                F = (k_eff(theta, deg = 'C0')*dolfin.inner(dolfin.grad(theta), dolfin.grad(theta_))*dx +
                     prm.rho/dt*(THETA*c_p_eff(theta,deg='C0')+(1-THETA)*c_p_eff(theta_k,deg='C0'))*(dolfin.inner(theta,theta_)-dolfin.inner(theta_k, theta_))*dx - sum(q_form))

                # Full THETA time discretization scheme:
                # F = (THETA*(k_eff(theta, deg = 'C0')*dolfin.inner(dolfin.grad(theta), dolfin.grad(theta_))*dx +
                #             prm.rho/dt*c_p_eff(theta,deg='C0')*(dolfin.inner(theta,theta_)-dolfin.inner(theta_k,theta_))*dx - sum(q_form)) +
                #      (1-THETA)*(k_eff(theta_k, deg = 'C0')*dolfin.inner(dolfin.grad(theta_k), dolfin.grad(theta_))*dx +
                #                 prm.rho/dt*c_p_eff(theta_k,deg='C0')*(dolfin.inner(theta, theta_) - dolfin.inner(theta_k,theta_))*dx - sum(q_form_k)))

                problem = dolfin.NonlinearVariationalProblem(F,theta,bcs=bc_form,J=dolfin.derivative(F,theta))
                    
                solver = dolfin.NonlinearVariationalSolver(problem)
                solver.parameters["newton_solver"]=NEWTON_PARAMS
                    
                return solver, theta, theta_k

            def stefan_form_cao():
                # Temperature transforming model (Cao,1991)
                
                (T,bcs,theta,_theta,theta_)=stefan_function_spaces()
                
                theta_k=dolfin.project(theta_analytic,T,solver_type="cg",preconditioner_type="hypre_amg")

                q_form, q_form_k, bc_form=stefan_boundary_values(theta_,bcs)

                # Cao formulation source term
                def s(theta, theta0=prm.theta_m, eps=em.EPS):
                    return dolfin.conditional(abs(theta-theta0)<eps,prm.cp_m*eps + prm.L_m/2,dolfin.conditional(theta>theta0,prm.cp_s*eps+prm.L_m,prm.cp_s*eps))

                # Fully implicit time discretization scheme
                F = k_eff(theta,deg='C0')*dolfin.inner(dolfin.grad(theta),dolfin.grad(theta_))*dx+prm.rho/dt*(c_p_eff(theta,deg='disC')*(theta-prm.theta_m)+s(theta)-c_p_eff(theta_k,deg='disC')*(theta_k-prm.theta_m)-s(theta_k))*theta_*dx-sum(q_form)

                # Full THETA time dicretization scheme
                # F = prm.rho/dt*(c_p_eff(theta,deg='disC')*(theta - prm.theta_m) + s(theta) - c_p_eff(theta_k,deg='disC')*(theta_k - prm.theta_m) - s(theta_k))*theta_*dx + (THETA*(k_eff(theta,deg='C0')*dolfin.inner(dolfin.grad(theta),dolfin.grad(theta_))*dx - sum(q_form)) + (1-THETA)*(k_eff(theta_k,deg='C0')*dolfin.inner(dolfin.grad(theta_k),dolfin.grad(theta_))*dx - sum(q_form_k)))
    
                problem = dolfin.NonlinearVariationalProblem(F,theta,bcs=bc_form,J=dolfin.derivative(F,theta))
                solver = dolfin.NonlinearVariationalSolver(problem)
                solver.parameters["newton_solver"]=NEWTON_PARAMS
                    
                return solver, theta, theta_k
            
            methodswitch = {
                'EHC':stefan_form_ehc,
                'TTM':stefan_form_cao
            }
            return methodswitch.get(method,method+" is not implemented. Consider 'EHC', or 'TTM'.")

        # Set timesets for simulation, data output and plotting
        sim_timeset, dat_timeset, plot_timeset=stefan_loop_timesets()

        # Define progress bar
        numprogresspoints=11
        
        idx_progresspoints=np.round(np.linspace(0,len(sim_timeset)-1,numprogresspoints)).astype(int)
        
        progressbar_timeset=sim_timeset[idx_progresspoints]

        # Set data for initial step
        theta_analytic.t = t_0
        stefan_form_update_previous(t_0)
        
        # Dictionary sim contains forms for particular methods:
        sim={}
        for method in methods:
            
            (solver, theta, theta_k)=stefan_problem_form(method)()
            
            sim[method]=(solver,theta,theta_k)
            sim[method][1].rename("Temperature by "+method,"theta_"+method)

            sim[method][1].set_allow_extrapolation(True)

            # Important for non-linear solver, sets initial guess for Newton:
            sim[method][1].assign(sim[method][2])

        # Create FunctionSpace for analytic solution projection:
        if sim:
            T=sim[methods[0]][1].function_space()
        else:
            T=stefan_function_spaces()[0]
            
        #-------------------------------------
        # Data files:

        # I. FEniCS data (ParaView):

        if SAVE_DAT:
            # Create HDF5 for data storing:
            data_hdf=dolfin.HDF5File(mesh.mpi_comm(),'./out/data/'+str(DIM)+'d/data.h5',"w")

            # Create XDMF file for ParaView visualization:
            data_xdmf=dolfin.XDMFFile(mesh.mpi_comm(),'./out/data/'+str(DIM)+'d/data_viz.xdmf')

            # Static mesh setting, lowers file size:
            data_xdmf.parameters['rewrite_function_mesh']=False
        
            # Common mesh for all stored functions, lowers file size:
            data_xdmf.parameters['functions_share_mesh']=True
        
        # II. Python data (matplotlib):

        data_py={"front_pos":{},
                 "temp_dist":{},
                 "disc_params":{},
                 "problem_params":{"r1":prm.R1,
                                   "r2":prm.R2,
                                   "q0":prm.q_0,
                                   "lambda":lambda_,
                                   "sim_timeset":sim_timeset,
                                   "formulation":BOUNDARY_FORMULATION
                 }
        }
        
        for method in sim:
            data_py["front_pos"][method]=[]

        # III. Optional:
        # Text file for double checking
        if SAVE_FRONT_POS_TXT:
            # Creating output file:
            output_file = 'out/data/'+str(DIM)+'d/data_front_pos.txt'
            file_front_pos = open(output_file, 'w')
            file_front_pos.write('t s_analytic s_EHC s_TTM\n')
            file_front_pos.write('- ---------- ----- -----\n')
        #--------------------------------------
        # Set epsilon (mollifying parameter)

        # Various types of Linf temp grad bound:

        # 1. analytic:
        def kappa_d(n):
            return n*np.pi**(n/2)/gamma(n/2+1)
        theta_grad_max_analytic=(prm.q_0*2**(1-DIM)/(prm.k_l*kappa_d(DIM)))*(lambda_**(1-DIM))*np.exp(-lambda_**2/prm.kappa_l)/np.sqrt(sim_timeset[0])
        #---------------------------------------

        # ------------------------------
        # Save discretization parameters
        # ------------------------------

        # Get h_eps and dt_cfl bound
        theta_0 = dolfin.project(theta_analytic,T,solver_type="cg",preconditioner_type="hypre_amg")
        
        global h_eps
        h_eps = em.get_h_eps(theta_0)

        global dt_cfl
        dt_cfl = em.get_delta_t_cfl(h_eps, lambda_/np.sqrt(t_0))
        
        # Into hdf file
        if SAVE_DAT:

            # Spatial discretization parameters:
            data_hdf.write(dolfin.project(hmax,T,solver_type="cg",preconditioner_type="hypre_amg"),"h_max")
            data_hdf.write(dolfin.project(hmin,T,solver_type="cg",preconditioner_type="hypre_amg"),"h_min")
            data_hdf.write(dolfin.project(prm.meshres[DIM],T,solver_type="cg",preconditioner_type="hypre_amg"),"meshres")

            # Temporal discretization parameters:
            data_hdf.write(dolfin.project(em.C_CFL,T,solver_type="cg",preconditioner_type="hypre_amg"),"C_CFL")
            data_hdf.write(dolfin.project(dt,T,solver_type="cg",preconditioner_type="hypre_amg"),"dt")

            # Temperature mollification parameters:
            data_hdf.write(dolfin.project(float(em.EPS),T,solver_type="cg",preconditioner_type="hypre_amg"),"eps")
            data_hdf.write(dolfin.project(h_eps,T,solver_type="cg",preconditioner_type="hypre_amg"),"h_eps")
            data_hdf.write(dolfin.project(em.C_EPS,T,solver_type="cg",preconditioner_type="hypre_amg"),"C_eps")            

        # Into numpy dict
        
        # Spatial discretization parameters:
        data_py["disc_params"]["h_max"] = hmax
        data_py["disc_params"]["h_min"] = hmin
        data_py["disc_params"]["meshres"] = prm.meshres[DIM]
        
        # Temporal discretization parameters:
        data_py["disc_params"]["dt"] = dt
        data_py["disc_params"]["C_CFL"] = em.C_CFL

        # Temperature mollification parameters:
        data_py["disc_params"]["eps"] = float(em.EPS)
        data_py["disc_params"]["h_eps"] = h_eps
        data_py["disc_params"]["C_eps"] = em.C_EPS
        # ==============================
        
        # Print information about simulation parameters:
        if rank==0:
            print(" ======================\n",
                  "Simulation parameters:\n",
                  "lambda = " + str(lambda_) + ",\n",
                  "Q_0 = " + str(prm.q_0) + ",\n",
                  "----------------------\n",
                  "Discretization parameters:\n",
                  "eps = " + str(float(em.EPS)) + ", (h_eps = " + str(h_eps) + " with C_eps = " + str(em.C_EPS) + "),\n",
                  "h_max = " + str(hmax) + ", h_min = " + str(hmin) + ",\n"
                  "dt = " + str(dt) + " (C_CFL = " + str(em.C_CFL) + '),\n',
                  "----------------------\n",
                  "Mesh parameters:\n",
                  "minimal edge length: " + str(minlength)+",\n",
                  "maximal edge length: " + str(maxlength)+",\n",
                  "average edge length: " + str(avlength)+",\n",
                  "number of cells: " + str(mesh.num_cells())+",\n",
                  "number of facets: " + str(mesh.num_facets())+",\n",
                  "number of edges: " + str(mesh.num_edges())+",\n",
                  "number of vertices: " + str(mesh.num_vertices())+",\n",
                  "======================\n",
            )

            # Print this to log file also
            if LOG:
                with open(log_filename,"a") as log_file:
                    log_file.write(" ======================\n"+
                                   "Simulation parameters:\n"+
                                   "lambda = " + str(lambda_) + ",\n"+
                                   "Q_0 = " + str(prm.q_0) + ",\n"+
                                   "----------------------\n"+
                                   "Discretization parameters:\n"+
                                   "eps = " + str(float(em.EPS)) + ", (h_eps = " + str(h_eps) + " with C_eps = " + str(em.C_EPS) + "),\n"+
                                   "h_max = " + str(hmax) + ", h_min = " + str(hmin) + ",\n"+
                                   "dt = " + str(dt) + " (C_CFL = " + str(em.C_CFL) + '),\n'+
                                   "----------------------\n"+
                                   "Mesh parameters:\n"+
                                   "minimal edge length = " + str(minlength)+",\n"+
                                   "maximal edge length = " + str(maxlength)+",\n"+
                                   "average edge length: " + str(avlength)+",\n"+
                                   "number of cells: " + str(mesh.num_cells())+",\n"+
                                   "number of facets: " + str(mesh.num_facets())+",\n"+
                                   "number of edges: " + str(mesh.num_edges())+",\n"+
                                   "number of vertices: " + str(mesh.num_vertices())+",\n"+
                                   "======================\n"
                    )
                    
        index = 0

        # ---------
        # Time loop
        # ---------
        
        for t in np.nditer(sim_timeset):
            
            # Update problem to current time level:
            stefan_form_update(t)

            # front pos txt:
            if SAVE_FRONT_POS_TXT:
                txt_row=str(t)+' '+str(2*lambda_*np.sqrt(t))

            # For melt front shape graph:
            # if index%10==0:
            #     fig, ax = plt.subplots(1,1)

            # Solve FEM problem for given methods:
            for method in sim:
                
                sim[method][0].solve()
                sim[method][2].assign(sim[method][1])

                # Front position calculation and export:
                front_position=stefan_front_position(sim[method][1])()
                data_py["front_pos"][method].append(front_position)

                if SAVE_FRONT_POS_TXT:
                    txt_row=txt_row+' '+str(front_position)

            if SAVE_FRONT_POS_TXT:
                txt_row=txt_row+'\n'
                file_front_pos.write(txt_row)

            # Update data to previous timestep:
            stefan_form_update_previous(t)

            # Data saving:
            if SAVE_DAT and (t in dat_timeset):

                # Save projection of analytic solution:
                theta_analytic_proj=dolfin.project(theta_analytic,T,solver_type="cg",preconditioner_type="hypre_amg")
                theta_analytic_proj.rename("Temperature analytic","theta_analytic")

                data_hdf.write(theta_analytic_proj,"theta_analytic",t)

                # t=t as a third argument is necessary due to pybind11 bug:
                data_xdmf.write(theta_analytic_proj,t=t)
                
                # Save temperature fields into data files:
                for method in sim:
                    data_hdf.write(sim[method][1],"theta_"+method,t)
                    data_xdmf.write(sim[method][1],t=t)

            # Save temp dist on a subdomain to npy data file:
            if t in plot_timeset:
                data_py["temp_dist"][str(t)]={"analytic":[[],[]]}
                x_range = np.arange(prm.R1,prm.R2,0.001)
                y_range = x_range*0.0
                for i,x in enumerate(x_range):
                    xp=dolfin.Point(x)
                    if bbox.compute_collisions(xp):
                        y_range[i]=theta_analytic(xp)
                if rank==0:
                    for process in range(comm.size)[1:]:
                        y=comm.recv(source=process)
                        y_range=np.maximum(y_range,y)
                    data_py["temp_dist"][str(t)]["analytic"][0]=x_range
                    data_py["temp_dist"][str(t)]["analytic"][1]=y_range
                else:
                    comm.send(y_range,dest=0)

                for method in sim:
                    y_range=0.0*x_range
                    data_py["temp_dist"][str(t)][method]=[[],[]]
                    for i,x in enumerate(x_range):
                        xp=dolfin.Point(x)
                        if bbox.compute_collisions(xp):
                            y_range[i]=sim[method][1](xp)
                        if rank==0:
                            for process in range(comm.size)[1:]:
                                y=comm.recv(source=process)
                                y_range=np.maximum(y_range,y)
                            data_py["temp_dist"][str(t)][method][0]=x_range
                            data_py["temp_dist"][str(t)][method][1]=y_range
                        else:
                            comm.send(y_range,dest=0)

            # Text file output (for double checking):
            if TEMP_TXT_DAT and (t in plot_timeset):
                # Creating output file:
                output_file = 'out/data/'+str(DIM)+'d/data_t_%s.txt' % (str(index))
                file_ = open(output_file, 'w')
                file_.write('x theta_analytic theta_EHC theta_TTM\n')
                file_.write('- -------------- --------- ---------\n')
            
                x_range = np.arange(prm.R1,prm.R2,mesh.hmin())
                for x in x_range:
                    row='{:f}'.format(x)+' '
                    p=dolfin.Point(x,0.,0.)
                    row=row+'{:f}'.format(theta_analytic(p))+' '
                    for method in sim:
                        row=row+'{:f}'.format(sim[method][1](p))+' '
                    row=row+'\n'
                    file_.write(row)
                file_.close()
                index=index+1

            # Progress bar
            if rank==0 and t in progressbar_timeset[1:]:
                progress=int((np.where(progressbar_timeset==t)[0])/(len(progressbar_timeset)-1)*100)
                print("Progress: "+str(progress)+"% complete")

                # Print to log txt file
                if LOG:
                    with open(log_filename,'a') as log_file:
                        log_file.write("Progress: "+str(progress)+"% complete\n")
                

        # END OF THE TIME LOOP
        #======================

        if SAVE_FRONT_POS_TXT:
            file_front_pos.close()

        # Save data dictionary for postprocessing:
        if rank==0:
            np.save('./out/data/'+str(DIM)+'d/data.npy', data_py)
        
        # Convergence data:
        if CONVERGENCE and sim:

            def eformat(f, prec, exp_digits):
                s = "%.*e"%(prec, f)
                mantissa, exp = s.split('e')
                # add 1 to digits as 1 is taken by sign +/-
                return "%se%+0*d"%(mantissa, exp_digits+1, int(exp))
            
            theta_analytic_proj=dolfin.project(theta_analytic,sim[method][1].function_space(),solver_type="cg",preconditioner_type="hypre_amg")
            errmethod={}
            for method in sim:
                front_position=data_py["front_pos"][method][-1]
                errmethod[method]=[dolfin.errornorm(theta_analytic_proj,sim[method][1],norm_type='L2')/dolfin.norm(theta_analytic_proj),
                                   dolfin.norm(theta_analytic_proj.vector()-sim[method][1].vector(),'linf')/dolfin.norm(theta_analytic_proj.vector(),'linf'),
                                   abs(front_position-2*lambda_*np.sqrt(dat_timeset[-1]))/2*lambda_*np.sqrt(dat_timeset[-1])
                ]
        
            # Write into file:
            if rank==0:
                with open('./out/data/'+str(DIM)+'d/convergence.csv', 'a') as csvfile:

                    filewriter = csv.writer(csvfile, delimiter=';',
                                            quoting=csv.QUOTE_NONE,
                    )
                    
                    params=r'\makecell{$\overline{\hcell}$=\numprint{'+eformat(avlength,1,1)+r'}, $\epsilon$=\numprint{'+eformat(float(em.EPS),1,1)+r'} \\ ($\delt$=\numprint{'+eformat(dt,1,1)+'})}'
                    for method in sim:
                        filewriter.writerow([params,
                                             method,
                                             r'\numprint{'+eformat(errmethod[method][0],2,1)+'}',
                                             r'\numprint{'+eformat(errmethod[method][1],2,1)+'}',
                                             r'\numprint{'+eformat(errmethod[method][2],2,1)+'}'
                        ])
                        params=''

        # Stability data:
        if STABILITY and sim:

            h=1/mesh.num_cells()
            eps = float(em.EPS)

            # Save discretization parameters:
            if not DATA_STABILITY['2p']['disc_params']:
                DATA_STABILITY['2p']['disc_params']['eps'] = eps
            elif not DATA_STABILITY['1p']['disc_params']:
                DATA_STABILITY['1p']['disc_params']['h'] = h
                DATA_STABILITY['1p']['disc_params']['dt'] = dt
            
            theta_analytic_proj=dolfin.project(theta_analytic,sim[method][1].function_space(),solver_type="cg",preconditioner_type="hypre_amg")
            for method in sim:
                fp=data_py["front_pos"][method][-1]
                fp_err=abs(front_position-2*lambda_*np.sqrt(sim_timeset[-1]))/(2*lambda_*np.sqrt(sim_timeset[-1]))
                
                l2_err=dolfin.errornorm(theta_analytic_proj,sim[method][1],norm_type='L2')/dolfin.norm(theta_analytic_proj)
                
                linf_err=dolfin.norm(theta_analytic_proj.vector()-sim[method][1].vector(),'linf')/dolfin.norm(theta_analytic_proj.vector(),'linf')
            try:    
                DATA_STABILITY['2p'][method][h][int(dt)]["fp_err"]=fp_err
                DATA_STABILITY['2p'][method][h][int(dt)]["l2_err"]=l2_err
                DATA_STABILITY['2p'][method][h][int(dt)]["linf_err"]=linf_err
            except KeyError:
                DATA_STABILITY['1p'][method][eps]["C_eps"]=h/h_eps
                DATA_STABILITY['1p'][method][eps]["fp_err"]=fp_err
                DATA_STABILITY['1p'][method][eps]["l2_err"]=l2_err
                DATA_STABILITY['1p'][method][eps]["linf_err"]=linf_err
        #==========================================================
        if SAVE_DAT:
            data_hdf.close()
            data_xdmf.close()
                
    return stefan_loop
# ================================================

# ---------------------------
# Various types of simulation
# ---------------------------

# Benchmark simulation:
def stefan_benchmark():
    
    # Preprocessing:
    (mesh,boundary,n,dx,ds)=smsh.stefan_mesh(DIM)()
    
    # Find analytic solution and lambda:
    (lambda_,theta_analytic,q_in,q_in_k,q_out,q_out_k)=stefan_analytic_sol(DIM)()
    
    # Compute FEM simulation:
    stefan_benchmark_sim(mesh,boundary,n,dx,ds,lambda_,theta_analytic,q_in,q_in_k,q_out,q_out_k,METHODS)()    

# Convergence simulation:
def stefan_convergence():

    em.C_CFL = 0.2
    
    convergence_params={
        1:{"meshres":[100,1000,10000],"eps":[5.0,0.5,0.05]},
        2:{"meshres":[31,155,775],"eps":[25.0,5.0,1.0]}, # this results in approx 10/100/1000 elements in a radius
        3:{"meshres":[0.05,0.025,0.0125],"eps":[25.,12.5,6.25]},
    }

    # Log into .txt file (cluster computing)
    global LOG
    LOG = True
    
    # Creating output log file:
    global log_filename
    log_filename = 'out/data/'+str(DIM)+'d/log.txt'
    log_file = open(log_filename, 'w')
    log_file.close()

    with open('./out/data/'+str(DIM)+'d/convergence.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=';',
                                quoting=csv.QUOTE_NONE,
        )
        
        # Write method type and value of L2, Linf, deltas error norm:
        filewriter.writerow(['params',
                             'method',
                             'l2norm',
                             'linfnorm',
                             'deltas'])

    for i,nx in enumerate(convergence_params[DIM]["meshres"]):
        prm.meshres[DIM]=nx
        em.EPS.assign(convergence_params[DIM]["eps"][i])
        stefan_benchmark()

        # Reset time-step
        global dt
        del dt

# One-parametric 1d stability benchmark:
def stability1p():

    global DATA_STABILITY    
    DATA_STABILITY = splt.load_data_stability()

    # Prepare data structure for one-parametric stability
    DATA_STABILITY['1p'] = {}
    DATA_STABILITY['1p']['disc_params'] = {}

    h = 1/prm.meshres[DIM]

    eps_range = [5., 4., 3., 2., 1.75, 1.5, 1.25, 1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]

    for method in ['EHC','TTM']:

        global METHODS
        METHODS = [method]

        # Make sure data for 2p stability will not be overwritten
        backup = DATA_STABILITY['2p'][method].pop(h, None)

        DATA_STABILITY['1p'][method] = {}
        
        for eps in eps_range:

            # Prepare dict for data output
            DATA_STABILITY['1p'][method][eps] = {}

            # Assign new value of epsilon
            em.EPS.assign(eps)

            # Compute benchmark problem for each method separately
            try:
                stefan_benchmark()
            except RuntimeError:
                DATA_STABILITY['1p'][method][eps]["C_eps"]=h/h_eps
                DATA_STABILITY['1p'][method][eps]["fp_err"]=1
                DATA_STABILITY['1p'][method][eps]["l2_err"]=1
                DATA_STABILITY['1p'][method][eps]["linf_err"]=1

        # Return data backup to 2p stability data structure
        if backup:
            DATA_STABILITY['2p'][method][h] = backup

    # Save disc parameters:
    DATA_STABILITY['1p']['disc_params']['h'] = h
    DATA_STABILITY['1p']['disc_params']['dt'] = dt
    DATA_STABILITY['1p']['disc_params']['C_CFL'] = em.C_CFL
            
    # Save stability data for postprocessing:
    if rank==0:
        np.save('./out/data/'+str(DIM)+'d/data_stability.npy', DATA_STABILITY) 

    
        
# Two-parametric stability benchmark:
def stability2p():

    global DATA_STABILITY
    DATA_STABILITY = splt.load_data_stability()

    # Prepare data structure for one-parametric stability
    DATA_STABILITY['2p'] = {}
    DATA_STABILITY['2p']['disc_params'] = {}

    global METHODS

    # for eps=5.0 is h_opt=1/100, deltat_opt=25000

    meshres=[1e1,2.5e1,5e1,7.5e1,1e2,2.5e2,5e2,7.5e2,1e3,2.5e3,5e3,7.5e3,1e4]
    timesteps=[1e2,2.5e2,5e2,7.5e2,1e3,2.5e3,5e3,7.5e3,1e4,2.5e4,5e4,7.e4,1e5]

    meshres=[]
    timesteps=[]
    for n in np.linspace(1,3,3):
        meshres=np.append(meshres,np.linspace(10**n,9*10**n,9))
        timesteps=np.append(timesteps,np.linspace(10**(n+1),9*10**(n+1),9))
    meshres=np.append(meshres,1e4)
    timesteps=np.append(timesteps,1e5)

    for method in ['EHC','TTM']:

        global METHODS
        METHODS = [method]

        DATA_STABILITY['2p'][method] = {}
        
        for nx in meshres:

            # Set new value of meshres
            prm.meshres[DIM]=int(nx)

            # Prepare dict for data output
            DATA_STABILITY['2p'][method][1/nx] = {}

            for deltat in timesteps:

                # Set new value of timestep
                global dt
                dt = deltat

                # Prepare dist for data output
                DATA_STABILITY['2p'][method][1/nx][dt] = {}

                try:
                    stefan_benchmark()
                except RuntimeError:
                    DATA_STABILITY['2p'][method][1/nx][dt]["fp_err"]=1
                    DATA_STABILITY['2p'][method][1/nx][dt]["l2_err"]=1
                    DATA_STABILITY['2p'][method][1/nx][dt]["linf_err"]=1

    # Save disc parameters
    DATA_STABILITY['2p']['disc_params']['eps'] = float(em.EPS)
    DATA_STABILITY['2p']['disc_params']['h_eps'] = h_eps
    DATA_STABILITY['2p']['disc_params']['dt_cfl'] = dt_cfl

    # Save stability data for postprocessing:
    if rank==0:
        np.save('./out/data/'+str(DIM)+'d/data_stability.npy', DATA_STABILITY)

# ---------
# Dev notes
# ---------
# Check difference btw project and interpolate
