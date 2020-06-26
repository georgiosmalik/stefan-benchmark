# ----------
# Source for Stefan benchmark simulation
# --------------------------------------

# Tento soubor obsahuje samotnou simulaci pomoci enthalpy metody
# Struktura:
# I. Analyticke reseni problemu:
# I.1. Vyresit transcendentni rovnici - vrati lambdu
# I.2. Definice presneho reseni - vrati expression pro reseni a okrajove podm

# II. FEM simulace
# II.1. Casove paramtery - cas zacatku a konce simulace -to da lambda + volim dt
# II.2. Vypocet ulohy - mam solver a funkce, vracim updatovane funkce
# II.3. Ukladam data

# Poznamky a opravy:
# DONE 1. pri generovani prostoru varianty pro ruzne okrajove podminky: Dirichlet/Neumann
# 2. prostor funkci muzu dostat z objektu DirichletBC, neni tedy treba ve stefan_functionspaces vracet zvlast function space a zvlast boundary_condition
# DONE 3. sloucit hledani lambdy a presne reseni?
# DONE (ANO) 4. lze ukladat vse do jednoho xdmf souboru?
# DONE 5. vykreslovani dat do samostatneho souboru postprocessingu?
# DONE 6. prirazeni okrajovek a toku podle schematu formulace-samostatna funkce
# DONE (zbytecne pro tento program) 7. MeshFunction, ktera umozni obarvovat hranice stringem, "inner","outer"
# DONE 8. Zajistit vytvoreni adresarove struktury pro ukladani dat
# DONE 9. Dodelat nelinearni formulace do definic forem
# 10. stefan_form_em,... jako objekty?

# Ukoly navic:
# 1. Zjistit proc nefunguje tvoje formulace LS.

# Chyby:
# zkontroluj c_p_eff, c_pm uz je obsazeno v mollify
# formulace ehc je spatne, nederivujes cp, ale jen teplotu
# DD formulace, spatne okrajovky


import dolfin
import numpy as np
import matplotlib.pyplot as plt
import csv

import pre.stefan_mesh as msh
import sim.params as prm
import sim.enthalpy_method as em
import post.my_plot as mplt
import post.stefan_plot as splt

from scipy.special import erf, erfc, expi, gamma, gammaincc
from scipy.optimize import fsolve
from math import floor, ceil

import time

# IMPORTANT FOR NON-LINEAR FORM----------
dolfin.parameters['form_compiler']['quadrature_degree']=8
# ---------------------------------------

dolfin.set_log_level(30)

#------------------------------------
# Global parameters of the simulation
#====================================
DIM=0
BOUNDARY_FORMULATION="DN"

# Flag for linear/nonlinear formulation of ehc and em
NONLINEAR=True

# Degree of finite element spaces:
DEGREE=1

# Starting and ending radius of simulation
R_START=0.4
R_END=0.65

# CFL condition constant
C_CFL=0.1

# Mollification constant
em.C_EPS=1.

# Number of timesteps, overridden by CFL condition
TIMESTEPS=1000

# Nonlinear solver parameters
NEWTON_PARAMS=dolfin.Parameters("newton_solver")
NEWTON_PARAMS.add("absolute_tolerance",1e-5)
NEWTON_PARAMS.add("maximum_iterations",25)
#NONLINEAR_SOLVER_PARAMS={"newton_solver":{"absolute_tolerance":1e-5,
                                          #"relaxation_parameter":1.0,
                                          #"maximum_iterations":100,
                                          #},
                         #}
#=====================================

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
            #f = lambda x : prm.rho*prm.L_m*np.sqrt(np.pi)*x + prm.k_l/np.sqrt(prm.kappa_l)*(prm.theta_m - prm.theta_0)/erf(x/np.sqrt(prm.kappa_l))*np.exp(-x*x/prm.kappa_l) + prm.k_s/np.sqrt(prm.kappa_s)*(prm.theta_m - prm.theta_inf)/erfc(x/np.sqrt(prm.kappa_s))*np.exp(-x*x/prm.kappa_s)

            # podle clanku:
            f=lambda x: prm.rho*prm.L_m*x-prm.q_0*np.exp(-x*x/prm.kappa_l)+prm.k_s/np.sqrt(prm.kappa_s*np.pi)*(prm.theta_m - prm.theta_inf)/erfc(x/np.sqrt(prm.kappa_s))*np.exp(-x*x/prm.kappa_s)
            lambda_ = fsolve(f,0.00001)
            if savefig:
                ax, _ = graph_transcendental_eq(lambda_,f,1)
                ax.axhline(y=0, lw=1, color='k')
            return lambda_[0]

        lambda_=transcendental_eq_1d(ploteq)

        # Analytic solution cpp code:
        #code_analytic="""x[0] < 2*lambda_*sqrt(t) ? theta_0 + (theta_m - theta_0)/erf(lambda_/sqrt(kappa_l))*erf(x[0]/(2*sqrt(t*kappa_l))) : theta_inf + (theta_m - theta_inf)/erfc(lambda_/sqrt(kappa_s))*erfc(x[0]/(2*sqrt(t*kappa_s)))"""

        # podle clanku:
        code_analytic="""x[0] < 2*lambda_*sqrt(t) ? q_0*sqrt(kappa_l*pi)/k_l*(erf(lambda_/sqrt(kappa_l))-erf(x[0]/(2*sqrt(kappa_l*t))))+theta_m : theta_inf + (theta_m - theta_inf)/erfc(lambda_/sqrt(kappa_s))*erfc(x[0]/(2*sqrt(t*kappa_s)))"""
        
        theta_analytic=dolfin.Expression(code_analytic,lambda_=lambda_, t=0.1, q_0=prm.q_0, theta_m=prm.theta_m, theta_0=prm.theta_0, theta_inf=prm.theta_inf, k_l=prm.k_l, kappa_l=prm.kappa_l, kappa_s=prm.kappa_s, degree=3)

        # cpp code heat flux:
        code_flux='-k*C1*exp(-r*r/(4*t*kappa))/sqrt(t)'

        # Heat influx:
        #q_in=dolfin.Expression(code_flux, t=0.1, k=prm.k_l, C1=(prm.theta_m-prm.theta_0)/(erf(lambda_/np.sqrt(prm.kappa_l))*np.sqrt(np.pi*prm.kappa_l)), r=prm.R1, kappa=prm.kappa_l, degree=0)

        # podle clanku:
        q_in=dolfin.Expression(code_flux, t=0.1, k=prm.k_l, C1=-prm.q_0/prm.k_l, r=prm.R1, kappa=prm.kappa_l, degree=0)

        # Heat outflux:
        q_out=dolfin.Expression(code_flux, t=0.1, k=prm.k_s, C1=(prm.theta_m-prm.theta_inf)/(erfc(lambda_/np.sqrt(prm.kappa_s))*np.sqrt(np.pi*prm.kappa_s)), r=prm.R2, kappa=prm.kappa_s, degree=0)
        
        return lambda_, theta_analytic, q_in, q_out

    def theta_sol_2d():

        def transcendental_eq_2d(savefig):
            
            f = lambda x : prm.rho*prm.L_m*x**2 - prm.k_l*prm.theta_0*np.exp(-(x**2)/(prm.kappa_l)) + prm.k_s*(prm.theta_m-prm.theta_inf)*np.exp(-(x**2)/prm.kappa_s)/(-expi(-(x**2)/prm.kappa_s))

            # podle clanku:
            f = lambda x : prm.rho*prm.L_m*x**2 - prm.q_0/(4*np.pi)*np.exp(-(x**2)/(prm.kappa_l)) - prm.k_s*(prm.theta_m-prm.theta_inf)*np.exp(-(x**2)/prm.kappa_s)/(expi(-(x**2)/prm.kappa_s))
            
            lambda_ = fsolve(f,0.00001)
            
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

          double t, theta_inf, theta_m, kappa_l, kappa_s, lambda_, c_2d;
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
               values[0] = theta_inf - (theta_inf - theta_m)/expint(-lambda_*lambda_/kappa_s)*expint(-f_s);
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
            .def_readwrite("theta_inf", &StefanAnalytic2d::theta_inf)
            .def_readwrite("t", &StefanAnalytic2d::t);
        }
        '''
        
        theta_analytic = dolfin.CompiledExpression(dolfin.compile_cpp_code(code_analytic).StefanAnalytic2d(),
                                      kappa_l=prm.kappa_l,
                                      kappa_s=prm.kappa_s,
                                      theta_m=prm.theta_m,
                                      c_2d=prm.theta_0,
                                      theta_inf=prm.theta_inf,
                                      lambda_=lambda_,
                                      t=0.1,
                                      degree=3)

        # podle clanku:
        theta_analytic = dolfin.CompiledExpression(dolfin.compile_cpp_code(code_analytic).StefanAnalytic2d(),
                                      kappa_l=prm.kappa_l,
                                      kappa_s=prm.kappa_s,
                                      theta_m=prm.theta_m,
                                      c_2d=prm.q_0/(4*np.pi*prm.k_l),
                                      theta_inf=prm.theta_inf,
                                      lambda_=lambda_,
                                      t=0.1,
                                      degree=3)

        # 2d heat flux cpp code:
        code_flux='-k*c_2d*2*exp(-r*r/(4*kappa*t))/r'

        # Heat influx:
        q_in=dolfin.Expression(code_flux, t=0.1, k=prm.k_l, c_2d=-prm.theta_0, r=prm.R1, kappa=prm.kappa_l, degree=0)

        # podle clanku:
        q_in=dolfin.Expression(code_flux, t=0.1, k=prm.k_l, c_2d=-prm.q_0/(2*np.pi*prm.k_l), r=prm.R1, kappa=prm.kappa_l, degree=0)

        # Heat outflux:
        q_out=dolfin.Expression(code_flux, t=0.1, k=prm.k_s, c_2d=-(prm.theta_m-prm.theta_inf)/expi(-lambda_**2/prm.kappa_s), r=prm.R2, kappa=prm.kappa_s, degree=0)
        return lambda_, theta_analytic, q_in, q_out

    def theta_sol_3d():

        def transcendental_eq_3d(savefig):
            
            f = lambda x : prm.rho*prm.L_m*x**3 - prm.k_l*np.sqrt(prm.kappa_l)*prm.theta_0*np.exp(-(x**2)/prm.kappa_l) + prm.k_s*np.sqrt(prm.kappa_s)*(prm.theta_m - prm.theta_inf)*np.exp(-(x**2)/prm.kappa_s)/((-2)*gamma(0.5)*gammaincc(0.5,x**2/prm.kappa_s) + 2*np.sqrt(prm.kappa_s)/x*np.exp(-x**2/prm.kappa_s))

            # podle clanku:
            f = lambda x : prm.rho*prm.L_m*x**3 - prm.q_0/(16*np.pi)*np.exp(-(x**2)/prm.kappa_l) + prm.k_s*np.sqrt(prm.kappa_s)*(prm.theta_m - prm.theta_inf)*np.exp(-(x**2)/prm.kappa_s)/((-2)*gamma(0.5)*gammaincc(0.5,x**2/prm.kappa_s) + 2*np.sqrt(prm.kappa_s)/x*np.exp(-x**2/prm.kappa_s))

            lambda_ = fsolve(f,0.00001)
            
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

          double t, theta_inf, theta_m, kappa_l, kappa_s, lambda_, c_3d;
          
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
               values[0] = theta_inf - (theta_inf - theta_m)/(-2*tgamma(0.5,lambda_*lambda_/kappa_s) + 2*sqrt(kappa_s)/lambda_*exp(-lambda_*lambda_/kappa_s))*(-2*tgamma(0.5,f_s) + 2*sqrt(1/f_s)*exp(-f_s));
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
            .def_readwrite("theta_inf", &StefanAnalytic3d::theta_inf)
            .def_readwrite("t", &StefanAnalytic3d::t);
        }
        '''

        # Compile cpp code for dolfin:
        theta_analytic = dolfin.CompiledExpression(dolfin.compile_cpp_code(code_analytic).StefanAnalytic3d(),
                                                   kappa_l=prm.kappa_l,
                                                   kappa_s=prm.kappa_s,
                                                   theta_m=prm.theta_m,
                                                   c_3d=prm.theta_0,
                                                   theta_inf=prm.theta_inf,
                                                   lambda_=lambda_,
                                                   t=0.1,
                                                   degree=3)

        # podle clanku:
        theta_analytic = dolfin.CompiledExpression(dolfin.compile_cpp_code(code_analytic).StefanAnalytic3d(),
                                                   kappa_l=prm.kappa_l,
                                                   kappa_s=prm.kappa_s,
                                                   theta_m=prm.theta_m,
                                                   c_3d=prm.q_0/(16*np.pi*np.sqrt(prm.kappa_l)*prm.k_l),
                                                   theta_inf=prm.theta_inf,
                                                   lambda_=lambda_,
                                                   t=0.1,
                                                   degree=3)

        # 3d heat flux cpp code:
        code_flux='-k*c_3d*(-4)*exp(-r*r/(4*kappa*t))*sqrt(kappa*t)/(r*r)'

        # Heat influx:
        q_in=dolfin.Expression(code_flux, t=0.1, k=prm.k_l, c_3d=prm.theta_0, r=prm.R1, kappa=prm.kappa_l, degree=0)

        # podle clanku:
        q_in=dolfin.Expression(code_flux, t=0.1, k=prm.k_l, c_3d=prm.q_0/(16*np.pi*np.sqrt(prm.kappa_l)*prm.k_l), r=prm.R1, kappa=prm.kappa_l, degree=0)

        # Heat outflux:
        q_out=dolfin.Expression(code_flux, t=0.1, k=prm.k_s, c_3d=-(prm.theta_m-prm.theta_inf)/((-2)*gamma(0.5)*gammaincc(0.5,lambda_**2/prm.kappa_s) + 2*np.sqrt(prm.kappa_s)/lambda_*np.exp(-lambda_**2/prm.kappa_s)), r=prm.R2, kappa=prm.kappa_s, degree=0)
        
        return lambda_, theta_analytic, q_in, q_out
    
    dimswitch = {
        1:theta_sol_1d,
        2:theta_sol_2d,
        3:theta_sol_3d
        }
    return dimswitch.get(dim, "Please enter 1d, 2d, or 3d.")

def stefan_benchmark_sim(mesh, boundary, n, dx, ds, lambda_, theta_analytic, q_in, q_out, methods):

    # MPI objects:
    comm=dolfin.MPI.comm_world
    bbox=mesh.bounding_box_tree()
    rank=dolfin.MPI.rank(mesh.mpi_comm())
    
    def stefan_loop_timesets():
        # Nastav pocatecni a koncovy cas podle presne polohy fronty tani:
        t_start=(R_START/(2*lambda_))**2
        t_end=(R_END/(2*lambda_))**2

        # Set timestep based on standart CFL condition:
        hmin=mesh.hmin()

        # Maximal velocity of melting front:
        vmax=lambda_/np.sqrt(t_start)

        global dt

        # Timestep given by CFL:
        dt=C_CFL*hmin/vmax

        # Timestep given by number of TIMESTEPS:
        #dt=int((t_end-t_start)/TIMESTEPS)

        # Vytvor timeset pro simulaci:
        sim_timeset=np.arange(t_start,t_end,dt)

        # Vytvor timeset pro ukladani:
        numdats=100

        if numdats >= len(sim_timeset):
            dat_timeset=sim_timeset
        else:
            idx_dats=np.round(np.linspace(0,len(sim_timeset)-1,numdats)).astype(int)
            dat_timeset=sim_timeset[idx_dats]

        # Vytvor timeset pro vykreslovani:
        numplots=3
        idx_plots=np.round(np.linspace(0,len(dat_timeset)-1,numplots)).astype(int)
        plot_timeset=dat_timeset[idx_plots]
        
        return sim_timeset, dat_timeset, plot_timeset

    def stefan_function_spaces(degree=DEGREE):
        # Tohle by melo vracet definice prostoru a funkce
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

    # Metoda na vypocet pozice fronty
    def stefan_front_position(theta):
        vol_ice=dolfin.assemble(em.mollify(1,0,theta,x0=prm.theta_m,eps=dolfin.DOLFIN_EPS,deg='disC')*dx)
        def front_pos_1d():
            return (prm.R2-vol_ice)
        def front_pos_2d():
            return (prm.R2**2-vol_ice/np.pi)**(1/2)
        def front_pos_3d():
            # Pocitame na siti, ktera je osminou koule
            return (prm.R2**3-6*vol_ice/np.pi)**(1/3)

        switch = {
            1:front_pos_1d,
            2:front_pos_2d,
            3:front_pos_3d
            }
        return switch.get(DIM)
    
    def stefan_loop():
        # Mel by vratit formulaci v zavislosti na okrajovych podminkach a pouzite metode (equiv heat capacity, cao - temp trans enthalpy method, nase metoda)
        
        def stefan_problem_form(method):
            # Specifikuje metodu, kterou se bude uloha pocitat

            def stefan_boundary_values(theta_tr,bc):
                # Vrat okrajove cleny v zavislosti na formulaci
                formulation={"DN":0,"ND":1,"NN":2,"DD":0.5}
                i=formulation[BOUNDARY_FORMULATION]
                q_form = [q_out*theta_tr*ds(2),q_in*theta_tr*ds(1)][floor(-1.5+i):ceil(0.5+i)]
                bc_form=bc[floor(0+i):ceil(1+i)]

                return q_form, bc_form

            # Define mollyfied parameters:
            def k_eff(theta,deg=em.DEG):
                return em.mollify(prm.k_s,prm.k_l,theta,x0=prm.theta_m,deg=deg)

            def c_p_eff(theta,deg=em.DEG):
                    return em.mollify(prm.cp_s,prm.cp_l,theta,x0=prm.theta_m,deg=deg)+em.dirac(prm.L_m,theta,x0=prm.theta_m,deg=deg)
                
            def stefan_form_em(nonlinear=NONLINEAR):
                # Vrat formulaci pro enthalpy method
                
                # Definuj prostor funkci:
                (T,bcs,theta,_theta,theta_)=stefan_function_spaces()
                
                # Nastav poc. podminku:
                theta_k=dolfin.project(theta_analytic,T)

                q_form, bc_form=stefan_boundary_values(theta_,bcs)

                # Nonlinear formulation:
                if nonlinear:
                    
                    F = k_eff(theta,deg='Cinf')*dolfin.inner(dolfin.grad(theta),dolfin.grad(theta_))*dx+prm.rho/dt*c_p_eff(theta,deg='Cinf')*(dolfin.inner(theta,theta_)-dolfin.inner(theta_k, theta_))*dx-sum(q_form)

                    problem = dolfin.NonlinearVariationalProblem(F,theta,bcs=bc_form,J=dolfin.derivative(F,theta))
                    solver = dolfin.NonlinearVariationalSolver(problem)
                    solver.parameters["newton_solver"]=NEWTON_PARAMS

                    return solver, theta, theta_k
                    
                # Linear formulation:
                F = k_eff(theta_k)*dolfin.inner(dolfin.grad(_theta), dolfin.grad(theta_))*dx+prm.rho*c_p_eff(theta_k)/dt*(dolfin.inner(_theta, theta_) - dolfin.inner(theta_k, theta_))*dx - sum(q_form)
                problem = dolfin.LinearVariationalProblem(dolfin.lhs(F),dolfin.rhs(F),theta,bc_form)
                solver = dolfin.LinearVariationalSolver(problem)
                
                return solver, theta, theta_k

            def stefan_form_ehc(nonlinear=NONLINEAR):
                # Vrat formulaci pro equivalent heat method
                
                # Definuj prostor funkci:
                (T,bcs,theta,_theta,theta_)=stefan_function_spaces()
                
                # Nastav poc. podminku:
                theta_k=dolfin.project(theta_analytic,T)

                q_form, bc_form=stefan_boundary_values(theta_,bcs)

                # Nonlinear formulation:
                if nonlinear:
                    
                    F = k_eff(theta,deg='C0')*dolfin.inner(dolfin.grad(theta),dolfin.grad(theta_))*dx+prm.rho/dt*c_p_eff(0.5*(theta+theta_k),deg='C0')*(dolfin.inner(theta,theta_)-dolfin.inner(theta_k, theta_))*dx-sum(q_form)

                    problem = dolfin.NonlinearVariationalProblem(F,theta,bcs=bc_form,J=dolfin.derivative(F,theta))
                    solver = dolfin.NonlinearVariationalSolver(problem)
                    solver.parameters["newton_solver"]=NEWTON_PARAMS

                    return solver, theta, theta_k    

                F = k_eff(theta_k,deg='C0')*dolfin.inner(dolfin.grad(_theta), dolfin.grad(theta_))*dx+prm.rho*c_p_eff(theta_k,deg='disC')/dt*(dolfin.inner(_theta, theta_) - dolfin.inner(theta_k, theta_))*dx - sum(q_form)
                
                problem = dolfin.LinearVariationalProblem(dolfin.lhs(F),dolfin.rhs(F),theta,bc_form)
                solver = dolfin.LinearVariationalSolver(problem)
                
                return solver, theta, theta_k

            def stefan_form_cao(nonlinear=True):
                # Temperature transforming model (Cao,1991)
                
                # Definuj prostor funkci:
                (T,bcs,theta,_theta,theta_)=stefan_function_spaces()
                
                # Nastav poc. podminku:
                theta_k=dolfin.project(theta_analytic,T)

                # Nastav okrajove cleny:
                q_form, bc_form=stefan_boundary_values(theta_,bcs)

                # Cao formulation source term
                def s(theta, theta0=prm.theta_m, eps=em.EPS):
                    return dolfin.conditional(abs(theta-theta0)<eps,prm.cp_m*eps + prm.L_m/2,dolfin.conditional(theta>theta0,prm.cp_s*eps+prm.L_m,prm.cp_s*eps))

                # Nonlinear formulation:
                F=k_eff(theta,deg='C0')*dolfin.inner(dolfin.grad(theta),dolfin.grad(theta_))*dx+prm.rho/dt*(c_p_eff(theta,deg='disC')*(theta-prm.theta_m)+s(theta)-c_p_eff(theta_k,deg='disC')*(theta_k-prm.theta_m)-s(theta_k))*theta_*dx-sum(q_form)
                
                problem = dolfin.NonlinearVariationalProblem(F,theta,bcs=bc_form,J=dolfin.derivative(F,theta))
                solver = dolfin.NonlinearVariationalSolver(problem)
                solver.parameters["newton_solver"]=NEWTON_PARAMS
                
                return solver, theta, theta_k
            
            methodswitch = {
                'em':stefan_form_em,
                'ehc':stefan_form_ehc,
                'cao':stefan_form_cao
            }
            return methodswitch.get(method,method+" is not implemented. Consider 'em', 'ehc', or 'cao'.")

        # Nastaveni casoveho schematu, plus startovniho casu pro analyt. res.
        sim_timeset, dat_timeset, plot_timeset=stefan_loop_timesets()
        theta_analytic.t=sim_timeset[0]
        
        # Dictionary sim contains forms for particular methods:
        sim={}
        for method in methods:
            
            (solver, theta, theta_k)=stefan_problem_form(method)()
            
            sim[method]=(solver,theta,theta_k)
            sim[method][1].rename("Temperature by "+method,"theta_"+method)

            # Important for non-linear solver, sets initial guess for Newton:
            sim[method][1].assign(sim[method][2])

        # Create FunctionSpace for analytic solution projection:
        if sim:
            T=sim[methods[0]][1].function_space()
        else:
            T=stefan_function_spaces()[0]
            
        #-------------------------------------
        # Data files:

        # I. Simulation results:
        
        # Create HDF5 for data storing:
        data_hdf=dolfin.HDF5File(mesh.mpi_comm(),'./out/data/'+str(DIM)+'d/data.h5',"w")

        # Create XDMF file for ParaView visualization:
        data_xdmf=dolfin.XDMFFile(mesh.mpi_comm(),'./out/data/'+str(DIM)+'d/data_viz.xdmf')

        # Static mesh setting, lowers file size:
        data_xdmf.parameters['rewrite_function_mesh']=False
        
        # Common mesh for all stored functions, lowers file size:
        data_xdmf.parameters['functions_share_mesh']=True
        
        # II. Melting front position:

        # Dict front_positions contains melting front positions:
        front_positions={}
        
        if rank==0:
            front_pos_file=open('out/data/'+str(DIM)+'d/front_pos','w+')

            for method in sim:
                front_pos_file.write('t, '+method+', ')
                front_positions[method]=[]
            front_pos_file.write('\n')
        #--------------------------------------
        # Set epsilon (mollifying parameter)
        em.set_eps(mesh,dolfin.project(theta_analytic,T))

        print('dt='+str(dt)+', eps='+str(em.EPS)+', nx='+str(prm.meshres)+', lambda='+str(lambda_)+', q_0='+str(prm.q_0))
        
        index=0
        #t0=time.time()
        # Time loop: 
        for t in np.nditer(sim_timeset):
            index=index+1
            # Update the analytical solution, form and boundary conditions:
            stefan_form_update(t)

            # Solve FEM problem for given methods:
            for method in sim:
                
                # Testing newton solver:
                #sim[method][1].assign(dolfin.project(theta_analytic,T))
                
                sim[method][0].solve()
                # if index==100:
                #     t1=time.time()
                #     print(t1-t0)
                #     exit()
                sim[method][2].assign(sim[method][1])

            # Data saving:
            if t in dat_timeset:

                # Save projection of analytic solution:
                theta_analytic_proj=dolfin.project(theta_analytic,T)
                theta_analytic_proj.rename("Temperature analytic","theta_analytic")

                data_hdf.write(theta_analytic_proj,"theta_analytic",t)

                # t=t as a third argument is necessary due to pybind11 bug:
                data_xdmf.write(theta_analytic_proj,t=t)

                #----------------------
                # Front pos testing:
                # front_pos_proj=dolfin.project(em.mollify(1,0,theta_analytic_proj,x0=prm.theta_m,eps=0.01,deg='disC'),T)
                # front_pos_proj.rename("Front position analytic","front_pos_analytic")

                # data_xdmf.write(front_pos_proj,t=t)
                #=======================
                
                # Uloz teplotni pole do datovych souboru:
                for method in sim:
                    data_hdf.write(sim[method][1],"theta_"+method,t)
                    data_xdmf.write(sim[method][1],t=t)
                
                    # Front position calculation and export:
                    front_position=stefan_front_position(sim[method][1])()
                    # if t in dat_timeset[3:]:
                    #     print(abs(front_position-2*lambda_*np.sqrt(t)))
                    #     p=dolfin.Point(front_position)
                    #     print(sim[method][1](p),theta_analytic(p))
                    #     temp2=dolfin.project(em.mollify(272,285,theta,x0=prm.theta_m,eps=0.4,deg='exact'),sim[method][1].function_space())
                    #     dolfin.plot(sim[method][1])
                    #     dolfin.plot(theta_analytic, mesh=mesh)
                    #     dolfin.plot(temp2)
                    #     plt.show()
                    #     exit()
                    if rank==0:
                        front_positions[method].append(front_position)

                        # Zapis hodnoty do dat souboru:
                        #front_pos_file.write('%.10f %.10f ' % (t,front_positions[method][-1]))
                # if rank==0:
                #     front_pos_file.write('\n')

        #-----------------------------------------------
        # Postprocessing:
        # Graph temp distribution along ray from origin:
        splt.DIM=DIM
        splt.graph_temp(dat_timeset,plot_timeset,theta_analytic,sim,data_hdf,comm,rank,bbox)

        # Graph position of the melting front:
        if rank==0:
            splt.graph_front_pos(dat_timeset,lambda_,front_positions)
            splt.graph_front_pos_diff(dat_timeset,lambda_,front_positions)
        #================================================
        
        # Vypisovani norem rozdilu do csv souboru:

        # Projekce presneho reseni:
        # if sim:
        #     theta_analytic_proj=dolfin.project(theta_analytic,sim[method][1].function_space())
        #     errmethod={}
        #     for method in sim:
        #         errmethod[method]=(dolfin.errornorm(theta_analytic_proj,sim[method][1],norm_type='L2')/dolfin.norm(theta_analytic_proj),
        #                            dolfin.norm(theta_analytic_proj.vector()-sim[method][1].vector(),'linf')/dolfin.norm(theta_analytic_proj.vector(),'linf'))
        
        #     # Zkontroluj vliv project/interpolation!!!
        
        #     # Vytvorime csv soubor na zapisovani:
        #     if rank==0:
        #         with open('./out/data/'+str(DIM)+'d/error_norms.csv', 'w') as csvfile:
        #             filewriter = csv.writer(csvfile, delimiter=',',
        #                                     quotechar='|',
        #                                     quoting=csv.QUOTE_MINIMAL)
        #             # Zapisujeme typ metody, L2 a L inf normu rel chyby
        #             filewriter.writerow(['method', 'l2norm', 'linfnorm'])
        #             for method in sim:
        #                 filewriter.writerow([method,
        #                                      str(errmethod[method][0]),
        #                                      str(errmethod[method][1])])
        #                 # Na posledni radek zapiseme parametry diskretizace
        #                 filewriter.writerow(['$\epsilon='+str(em.eps)+'$',
        #                                      '$nx='+str(prm.meshres)+'$',
        #                                      '$dt='+str(dt)+'$'] )
        # konec sekce pro vypisovani chyb do csv filu
        #==========================================================

        #================================================
        
        # Vypisovani dat pro konvergencni tabulku:

        # Projekce presneho reseni:
        if sim:
            theta_analytic_proj=dolfin.project(theta_analytic,sim[method][1].function_space())
            errmethod={}
            for method in sim:
                errmethod[method]=[dolfin.errornorm(theta_analytic_proj,sim[method][1],norm_type='L2')/dolfin.norm(theta_analytic_proj),
                                   dolfin.norm(theta_analytic_proj.vector()-sim[method][1].vector(),'linf')/dolfin.norm(theta_analytic_proj.vector(),'linf'),
                                   abs(front_positions[method][-1]-2*lambda_*np.sqrt(dat_timeset[-1]))/2*lambda_*np.sqrt(dat_timeset[-1])
                ]
        
            # Zkontroluj vliv project/interpolation!!!
        
            # Vytvorime csv soubor na zapisovani:
            if rank==0:
                with open('./out/data/'+str(DIM)+'d/convergence.csv', 'a') as csvfile:

                    filewriter = csv.writer(csvfile, delimiter=',',
                                            quotechar='|',
                                            quoting=csv.QUOTE_MINIMAL
                    )
                    
                    params='\# cells='+str(mesh.num_cells())+' (h='+'{0:.3f}'.format(mesh.hmax())+') dt='+'{0:>2.3e}'.format(dt)
                    for method in sim:
                        filewriter.writerow([params,
                                             method,
                                             '{0:>2.3e}'.format(errmethod[method][0]),
                                             '{0:>2.3e}'.format(errmethod[method][1]),
                                             '{0:>2.3e}'.format(errmethod[method][2])
                        ])
                        params=''
        # konec sekce pro vypisovani chyb do csv filu
        #==========================================================
                
    return stefan_loop

# Spust simulaci
def stefan_benchmark():
    # preprocessing:
    (mesh,boundary,n,dx,ds)=msh.stefan_mesh(DIM)()
    # find lambda and return analytic solution:
    (lambda_,theta_analytic,q_in,q_out)=stefan_analytic_sol(DIM)()
    # do the computation:
    stefan_benchmark_sim(mesh,boundary,n,dx,ds,lambda_,theta_analytic,q_in,q_out,['cao','ehc'])()

# Poznamka z 7.6.: bohuzel nelinearni formulace nekonverguje pro Cinf aproximace heavisida a diraca u cp_eff, pokud zvolis C1/Cinf pro HS/D pak to funguje, pro Cinf/Cinf haze nan pro reziduum, prozkoumej co se deje

# Poznamka z 8.6.: problem je v Cinf aproximaci heaviside funkce
