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

# IMPORTANT FOR NON-LINEAR FORM----------
dolfin.parameters['form_compiler']['quadrature_degree']=4
# ---------------------------------------

dolfin.set_log_level(30)

#------------------------------------
# Global parameters of the simulation
#====================================
DIM=0
BOUNDARY_FORMULATION="NN"
METHODS=['EHC','TTM']

# Flag for linear/nonlinear formulation of ehc and em
NONLINEAR=True

# Degree of finite element spaces:
DEGREE=1

# Starting and ending radius of simulation
R_START=0.4
R_END=0.65

# CFL condition constant
C_CFL=0.8

# Mollification constant
em.C_EPS=2.5

# Nonlinear solver parameters
NEWTON_PARAMS=dolfin.Parameters("newton_solver")
NEWTON_PARAMS.add("linear_solver","bicgstab")
NEWTON_PARAMS.add("absolute_tolerance",1e-5)
NEWTON_PARAMS.add("maximum_iterations",25)

# Specify data output
GRAPH=False
SAVE_DAT=False
TEMP_TXT_DAT=False
SAVE_FRONT_POS_TXT=True

# Types of simulation
BENCHMARK=False
CONVERGENCE=False
STABILITY=False

# Data for stability testing:
DATA_STABILITY={}

# Temporal scheme for EHC model
THETA=0.5

# TEST (tuning 3d benchmark)
PROJECT=False
#---------------------------
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

        # Heat outflux:
        q_out=dolfin.Expression(code_flux, t=0.1, k=prm.k_s, C1=(prm.theta_m-prm.theta_i)/(erfc(lambda_/np.sqrt(prm.kappa_s))*np.sqrt(np.pi*prm.kappa_s)), r=prm.R2, kappa=prm.kappa_s, degree=0)
        
        return lambda_, theta_analytic, q_in, q_out

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

        # Heat outflux:
        q_out=dolfin.Expression(code_flux, t=0.1, k=prm.k_s, c_2d=(-2)*(prm.theta_m-prm.theta_i)/expi(-lambda_**2/prm.kappa_s), r=prm.R2, kappa=prm.kappa_s, degree=0)
        return lambda_, theta_analytic, q_in, q_out

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

        # TEST (tuning 3d benchmark)
        if PROJECT:
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
                double f_l = (x[0]*x[0])/(4*kappa_l*t) ;
                double f_s = (x[0]*x[0])/(4*kappa_s*t) ;
                if ( x[0] <= 2*lambda_*sqrt(t) ) {
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
        #---------------------------

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

        # Heat outflux:
        q_out=dolfin.Expression(code_flux, t=0.1, k=prm.k_s, c_3d=-(prm.theta_i-prm.theta_m)/((-1)*gamma(0.5)*gammaincc(0.5,lambda_**2/prm.kappa_s) + np.sqrt(prm.kappa_s)/lambda_*np.exp(-lambda_**2/prm.kappa_s)), r=prm.R2, kappa=prm.kappa_s, degree=0)
        
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

    global rank
    rank=dolfin.MPI.rank(mesh.mpi_comm())
    
    def stefan_loop_timesets():

        # Nastav pocatecni a koncovy cas podle presne polohy fronty tani:
        t_0=(R_START/(2*lambda_))**2
        t_max=(R_END/(2*lambda_))**2

        # Set timestep based on standart CFL condition:

        global hmin
        
        hmin=dolfin.MPI.min(mesh.mpi_comm(),mesh.hmin())

        global hmax

        hmax=dolfin.MPI.max(mesh.mpi_comm(),mesh.hmax())

        # Maximal velocity of melting front:
        vmax=lambda_/np.sqrt(t_0)

        if BENCHMARK:
            global dt

            # Timestep given by CFL:
            dt=C_CFL*hmin/vmax

        print(dt*vmax/hmin)
        exit()

        # Vytvor timeset pro simulaci:
        sim_timeset=np.arange(t_0,t_max,dt)

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
        #vol_ice=dolfin.assemble(em.mollify(1,0,theta,x0=prm.theta_m,eps=dolfin.DOLFIN_EPS,deg='disC')*dx)
        vol_ice=dolfin.assemble(em.mollify(1,0,theta-prm.theta_m,x0=0,eps=em.EPS,deg='C0')*dx)
        
        def front_pos_1d():
            return prm.R2-vol_ice
        def front_pos_2d():
            return np.sqrt(prm.R2**2-vol_ice/np.pi)
        def front_pos_3d():
            # Pocitame na siti, ktera je osminou koule
            return np.cbrt(prm.R2**3-6*vol_ice/np.pi)

        # TEST (tuning 3d benchmark):
        if PROJECT:
            def front_pos_3d1d():
                return prm.R2-vol_ice
        #----------------------------

        switch = {
            1:front_pos_1d,
            2:front_pos_2d,
            3:front_pos_3d
        }
        
        # TEST (tuning 3d benchmark):
        if PROJECT:
            switch = {
                1:front_pos_1d,
                2:front_pos_2d,
                3:front_pos_3d1d
            }
        #----------------------------
        return switch.get(DIM)
    
    def stefan_loop():
        # Mel by vratit formulaci v zavislosti na okrajovych podminkach a pouzite metode (equiv heat capacity, TTM - temp trans enthalpy method, nase metoda)
        
        def stefan_problem_form(method):
            # Specifikuje metodu, kterou se bude uloha pocitat

            def stefan_boundary_values(theta_test,bc):
                # Vrat okrajove cleny v zavislosti na formulaci
                formulation={"DN":0,"ND":1,"NN":2,"DD":0.5}
                i=formulation[BOUNDARY_FORMULATION]
                
                q_form = [q_out*theta_test*ds(2),q_in*theta_test*ds(1)][floor(-1.5+i):ceil(0.5+i)]

                # TEST (tuning 3d benchmark)
                if PROJECT:
                    q_form = [q_out*theta_test*(prm.R2**2)*ds(2),q_in*theta_test*(prm.R1**2)*ds(1)][floor(-1.5+i):ceil(0.5+i)]
                #---------------------------
                
                bc_form=bc[floor(0+i):ceil(1+i)]

                return q_form, bc_form

            # Define mollified parameters:
            def k_eff(theta,deg=em.DEG):
                return em.mollify(prm.k_s,prm.k_l,theta,x0=prm.theta_m,deg=deg)

            def c_p_eff(theta,deg=em.DEG):
                    return em.mollify(prm.cp_s,prm.cp_l,theta,x0=prm.theta_m,deg=deg)+em.dirac(prm.L_m,theta,x0=prm.theta_m,deg=deg)
                
            def stefan_form_em(nonlinear=NONLINEAR):
                # Vrat formulaci pro enthalpy method
                
                # Definuj prostor funkci:
                (T,bcs,theta,_theta,theta_)=stefan_function_spaces()
                
                # Nastav poc. podminku:
                theta_k=dolfin.project(theta_analytic,T,solver_type="cg",preconditioner_type="hypre_amg")

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
                theta_k=dolfin.project(theta_analytic,T,solver_type="cg",preconditioner_type="hypre_amg")

                q_form, bc_form=stefan_boundary_values(theta_,bcs)

                # Nonlinear formulation:
                if nonlinear:
                    
                    F = k_eff(theta,deg='C0')*dolfin.inner(dolfin.grad(theta),dolfin.grad(theta_))*dx+prm.rho/dt*(THETA*c_p_eff(theta,deg='C0')+(1-THETA)*c_p_eff(theta_k,deg='C0'))*(dolfin.inner(theta,theta_)-dolfin.inner(theta_k, theta_))*dx-sum(q_form)

                    # TEST (tuning 3d benchmark)
                    if PROJECT:
                        jr = dolfin.Expression("x[0]*x[0]",domain = mesh, degree=3)
                        F = k_eff(theta,deg='C0')*dolfin.inner(dolfin.grad(theta),dolfin.grad(theta_))*jr*dx+prm.rho/dt*(THETA*c_p_eff(theta,deg='C0')+(1-THETA)*c_p_eff(theta_k,deg='C0'))*(dolfin.inner(theta,theta_)-dolfin.inner(theta_k, theta_))*jr*dx-sum(q_form)
                    #---------------------------

                    problem = dolfin.NonlinearVariationalProblem(F,theta,bcs=bc_form,J=dolfin.derivative(F,theta))
                    solver = dolfin.NonlinearVariationalSolver(problem)
                    solver.parameters["newton_solver"]=NEWTON_PARAMS
                    # solver.parameters["nonlinear_solver"]="newton"
                    # solver.parameters["newton_solver"]["linear_solver"]="bicgstab"
                    # solver.parameters["newton_solver"]["absolute_tolerance"]=1e-5
                    # solver.parameters["newton_solver"]["maximum_iterations"]=25
                    #solver.parameters["newton_solver"]["linear_solver"]="mumps"
                    #solver.parameters["newton_solver"]["linear_solver"]="cg" 
                    #solver.parameters["newton_solver"]["preconditioner"]="hypre_amg"
                    
                    
                    #solver.parameters["newton_solver"]['linear_solver'] = dolfin.Parameters("CG")
                    #dolfin.info(solver.parameters, True)
                    #exit()
                    #dolfin.info(solver.parameters["newton_solver"], True)
                    #solver.parameters["krylov_solver"]="CG"
                    #solver.parameters["newton_solver"]["linear_solver"]=dolfin.Parameters("CG")
                    #solver.parameters['snes_solver']['linear_solver']= "cg"
                    #solver.parameters['snes_solver']['preconditioner'] = "hypre_amg"
                    #exit()

                    

               
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
                theta_k=dolfin.project(theta_analytic,T,solver_type="cg",preconditioner_type="hypre_amg")

                # Nastav okrajove cleny:
                q_form, bc_form=stefan_boundary_values(theta_,bcs)

                # Cao formulation source term
                def s(theta, theta0=prm.theta_m, eps=em.EPS):
                    return dolfin.conditional(abs(theta-theta0)<eps,prm.cp_m*eps + prm.L_m/2,dolfin.conditional(theta>theta0,prm.cp_s*eps+prm.L_m,prm.cp_s*eps))

                # Nonlinear formulation:
                F=k_eff(theta,deg='C0')*dolfin.inner(dolfin.grad(theta),dolfin.grad(theta_))*dx+prm.rho/dt*(c_p_eff(theta,deg='disC')*(theta-prm.theta_m)+s(theta)-c_p_eff(theta_k,deg='disC')*(theta_k-prm.theta_m)-s(theta_k))*theta_*dx-sum(q_form)

                # TEST (tuning 3d benchmark)
                if PROJECT:
                    jr = dolfin.Expression("x[0]*x[0]",domain = mesh,degree=3)
                    F=k_eff(theta,deg='C0')*dolfin.inner(dolfin.grad(theta),dolfin.grad(theta_))*jr*dx+prm.rho/dt*(c_p_eff(theta,deg='disC')*(theta-prm.theta_m)+s(theta)-c_p_eff(theta_k,deg='disC')*(theta_k-prm.theta_m)-s(theta_k))*theta_*jr*dx-sum(q_form)
                #----------------------------
                
                problem = dolfin.NonlinearVariationalProblem(F,theta,bcs=bc_form,J=dolfin.derivative(F,theta))
                solver = dolfin.NonlinearVariationalSolver(problem)
                solver.parameters["newton_solver"]=NEWTON_PARAMS
                    
                return solver, theta, theta_k
            
            methodswitch = {
                'EHCreg':stefan_form_em,
                'EHC':stefan_form_ehc,
                'TTM':stefan_form_cao
            }
            return methodswitch.get(method,method+" is not implemented. Consider 'EHCreg', 'EHC', or 'TTM'.")

        # Nastaveni casoveho schematu, plus startovniho casu pro analyt. res.
        sim_timeset, dat_timeset, plot_timeset=stefan_loop_timesets()
        theta_analytic.t=sim_timeset[0]
        
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
        # 2. local:
        theta = dolfin.project(theta_analytic,T,solver_type="cg",preconditioner_type="hypre_amg")
        char = dolfin.conditional(abs(theta-prm.theta_m)<1.,1.,0.)
        theta_norm=dolfin.project(char*dolfin.sqrt(dolfin.inner(dolfin.grad(theta),dolfin.grad(theta))),T,solver_type="cg",preconditioner_type="hypre_amg")
        theta_grad_max_local=theta_norm.vector().norm('linf')

        # 3. global:
        theta_norm=dolfin.project(dolfin.sqrt(dolfin.inner(dolfin.grad(theta),dolfin.grad(theta))),theta.function_space())
        theta_grad_max_global=theta_norm.vector().norm('linf')

        # Experiment with local bound temp width
        # print(theta_grad_max_analytic, theta_grad_max_local, theta_grad_max_global)

        # em.set_eps(hmax,theta_grad_max_analytic)
        # print(float(em.EPS))
        # em.set_eps(hmax,theta_grad_max_local)
        # print(float(em.EPS))
        # em.set_eps(hmax,theta_grad_max_global)
        # print(float(em.EPS))

        # exit()

        # print("eps: ",float(em.EPS))
        # h_opt=em.C_EPS*2*float(em.EPS)/theta_grad_max_local
        # print("h pro C_EPS=",em.C_EPS,": ",h_opt)
        # print("dt pro C_CFL=",C_CFL,": ",C_CFL*h_opt*np.sqrt(sim_timeset[0])/lambda_)
        # exit()
        #---------------------------------------

        if BENCHMARK:
            em.set_eps(hmax,theta_grad_max_local)

        print('dt='+str(dt)+', eps='+str(float(em.EPS))+', h_max='+str(hmax)+', lambda='+str(lambda_)+', q_0='+str(prm.q_0))

        # Save discretization parameters:
        if SAVE_DAT:
            data_hdf.write(dolfin.project(em.C_EPS,T,solver_type="cg",preconditioner_type="hypre_amg"),"C_eps")
            data_hdf.write(dolfin.project(C_CFL,T,solver_type="cg",preconditioner_type="hypre_amg"),"C_CFL")
            data_hdf.write(dolfin.project(hmax,T,solver_type="cg",preconditioner_type="hypre_amg"),"h_max")
            data_hdf.write(dolfin.project(hmin,T,solver_type="cg",preconditioner_type="hypre_amg"),"h_min")

        data_py["disc_params"]["C_eps"]=em.C_EPS
        data_py["disc_params"]["C_CFL"]=C_CFL
        data_py["disc_params"]["h_max"]=hmax
        data_py["disc_params"]["h_min"]=hmin
        data_py["disc_params"]["eps"]=float(em.EPS)
        data_py["disc_params"]["dt"]=dt
        data_py["disc_params"]["meshres"]=prm.meshres
        
        index=0
        
        # Time loop: 
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
                    

            # -----------------------------
            # Graph shape of the melt front
            #     if index%10==0:
            #         dolfin.plot(sim[method][1])

            # if index<=50 and index%10==0:
            #     print(index)
            #     theta_analytic_proj=dolfin.project(theta_analytic,T)
            #     dolfin.plot(theta_analytic_proj)
            #     ax.set_ylim(272.75, 273.5)
            #     ax.set_xlim(2*lambda_*np.sqrt(t)-0.005,2*lambda_*np.sqrt(t)+0.005)
            #     plt.show()


            # index=index+1

            # if index==51:
            #     exit()
            #=================================

            # Data saving:
            if SAVE_DAT and (t in dat_timeset):

                # Save projection of analytic solution:
                theta_analytic_proj=dolfin.project(theta_analytic,T,solver_type="cg",preconditioner_type="hypre_amg")
                theta_analytic_proj.rename("Temperature analytic","theta_analytic")

                data_hdf.write(theta_analytic_proj,"theta_analytic",t)

                # t=t as a third argument is necessary due to pybind11 bug:
                data_xdmf.write(theta_analytic_proj,t=t)
                
                # Uloz teplotni pole do datovych souboru:
                for method in sim:
                    data_hdf.write(sim[method][1],"theta_"+method,t)
                    data_xdmf.write(sim[method][1],t=t)

            # Uloz do python data filu teplotni pole (paralelne):
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

            # Kod na vytvareni binarnich datovych souboru
            if TEMP_TXT_DAT and (t in plot_timeset):
                # Creating output file:
                output_file = 'out/data/'+str(DIM)+'d/data_t_%s.txt' % (str(index))
                file_ = open(output_file, 'w')
                file_.write('x theta_analytic theta_EHC theta_TTM\n')
                file_.write('- -------------- --------- ---------\n')
            
                # polomer se rozdeli na intervaly delky hmin
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
        #======================

        if SAVE_FRONT_POS_TXT:
            file_front_pos.close()

        # Save data dictionary for postprocessing:
        if rank==0:
            np.save('./out/data/'+str(DIM)+'d/data.npy', data_py)
        #----------------------
        # Visual postprocessing
        #======================

        # 1d graphs:
        if GRAPH and rank==0:
            splt.load_data()
            
            # Graph temp distribution along ray from origin:
            splt.graph_temp()
            
            # Graph position of the melting front:
            splt.graph_front_pos()

            # Graph velocity of the melting front:
            splt.graph_front_vel()
        
        # Convergence data:
        if CONVERGENCE and sim:
            theta_analytic_proj=dolfin.project(theta_analytic,sim[method][1].function_space(),solver_type="cg",preconditioner_type="hypre_amg")
            errmethod={}
            for method in sim:
                front_position=data_py["front_pos"][method][-1]
                errmethod[method]=[dolfin.errornorm(theta_analytic_proj,sim[method][1],norm_type='L2')/dolfin.norm(theta_analytic_proj),
                                   dolfin.norm(theta_analytic_proj.vector()-sim[method][1].vector(),'linf')/dolfin.norm(theta_analytic_proj.vector(),'linf'),
                                   abs(front_position-2*lambda_*np.sqrt(dat_timeset[-1]))/2*lambda_*np.sqrt(dat_timeset[-1])
                ]
            
            # Zkontroluj vliv project/interpolation!!!
        
            # Write into file:
            if rank==0:
                with open('./out/data/'+str(DIM)+'d/convergence.csv', 'a') as csvfile:

                    filewriter = csv.writer(csvfile, delimiter=',',
                                            #quotechar='|',
                                            quoting=csv.QUOTE_MINIMAL
                    )
                    
                    params=r'\makecell{$h$=\numprint{'+'{0:>2.1e}'.format(hmax)+r'}, $\epsilon$=\numprint{'+'{0:>2.1e}'.format(float(em.EPS))+r'} \\ ($\Delta t$=\numprint{'+r'{0:>2.1e}'.format(dt)+'})}'
                    for method in sim:
                        filewriter.writerow([params,
                                             method,
                                             r'\numprint{'+'{0:>2.2e}'.format(errmethod[method][0])+'}',
                                             r'\numprint{'+'{0:>2.2e}'.format(errmethod[method][1])+'}',
                                             r'\numprint{'+'{0:>2.2e}'.format(errmethod[method][2])+'}'
                        ])
                        params=''

        # Stability data:
        if STABILITY and sim:
            theta_analytic_proj=dolfin.project(theta_analytic,sim[method][1].function_space(),solver_type="cg",preconditioner_type="hypre_amg")
            h=1/mesh.num_cells()
            for method in sim:
                fp=data_py["front_pos"][method][-1]
                fp_err=abs(front_position-2*lambda_*np.sqrt(sim_timeset[-1]))/(2*lambda_*np.sqrt(sim_timeset[-1]))
                
                l2_err=dolfin.errornorm(theta_analytic_proj,sim[method][1],norm_type='L2')/dolfin.norm(theta_analytic_proj)
                
                linf_err=dolfin.norm(theta_analytic_proj.vector()-sim[method][1].vector(),'linf')/dolfin.norm(theta_analytic_proj.vector(),'linf')
                
                DATA_STABILITY[h][int(dt)][method]["fp_err"]=fp_err
                DATA_STABILITY[h][int(dt)][method]["l2_err"]=l2_err
                DATA_STABILITY[h][int(dt)][method]["linf_err"]=linf_err
        #==========================================================
        if SAVE_DAT:
            data_hdf.close()
            data_xdmf.close()
                
    return stefan_loop

# Spust simulaci
def stefan_benchmark():
    # preprocessing:
    (mesh,boundary,n,dx,ds)=smsh.stefan_mesh(DIM)()
    # find lambda and return analytic solution:
    (lambda_,theta_analytic,q_in,q_out)=stefan_analytic_sol(DIM)()
    # do the computation:
    stefan_benchmark_sim(mesh,boundary,n,dx,ds,lambda_,theta_analytic,q_in,q_out,METHODS)()

# TEST (tuning 3d benchmark)
def stefan_projection():
    # preprocessing:
    (mesh,boundary,n,dx,ds)=smsh.stefan_mesh(1)()
    # find lambda and return analytic solution:
    (lambda_,theta_analytic,q_in,q_out)=stefan_analytic_sol(DIM)()
    # do the computation:
    stefan_benchmark_sim(mesh,boundary,n,dx,ds,lambda_,theta_analytic,q_in,q_out,['EHC','TTM'])()
#---------------------------
    

# Convergence simulation:
def stefan_convergence():
    meshres={
    1:[100,1000,10000],
    2:[25],
    3:[0.05,0.025,0.01]
    }

    with open('./out/data/'+str(DIM)+'d/convergence.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)

        # Zapisujeme typ metody, L2 a L inf normu rel chyby
        filewriter.writerow(['params',
                             'method',
                             'l2norm',
                             'linfnorm',
                             'deltas'])

    for i,nx in enumerate(meshres[DIM]):
        prm.meshres[DIM]=nx
        global dt
        dt = 2*meshres[DIM][-i-1]
        em.EPS.assign(50./(10**(i+1)))
        stefan_benchmark()

# Stability benchmark:
def stefan_stability():
    # pro pevne epsilon napocita chyby pozice fronty pro prislusne volby casoveho kroku a prostoroveho kroku

    global DATA_STABILITY

    global METHODS

    # Compute with fixed epsilon:
    em.EPS.assign(1.25)

    # for eps=5.0 is h_opt=1/100, deltat_opt=25000

    meshres=[1e1,2.5e1,5e1,7.5e1,1e2,2.5e2,5e2,7.5e2,1e3,2.5e3,5e3,7.5e3,1e4]
    timesteps=[1e2,2.5e2,5e2,7.5e2,1e3,2.5e3,5e3,7.5e3,1e4,2.5e4,5e4,7.5e4,1e5]

    meshres=[]
    timesteps=[]
    for n in np.linspace(1,3,3):
        meshres=np.append(meshres,np.linspace(10**n,9*10**n,9))
        timesteps=np.append(timesteps,np.linspace(10**(n+1),9*10**(n+1),9))
    meshres=np.append(meshres,1e4)
    timesteps=np.append(timesteps,1e5)
    
    for nx in meshres:
        prm.meshres[DIM]=int(nx)
        DATA_STABILITY[1/nx]={}
        for deltat in timesteps:
            global dt
            dt = deltat
            DATA_STABILITY[1/nx][dt]={}
            for method in ['EHC','TTM']:
                DATA_STABILITY[1/nx][dt][method]={}
                global METHODS
                METHODS=[method]
                try:
                    stefan_benchmark()
                except RuntimeError:
                    DATA_STABILITY[1/nx][dt][method]["fp_err"]=1
                    DATA_STABILITY[1/nx][dt][method]["l2_err"]=1
                    DATA_STABILITY[1/nx][dt][method]["linf_err"]=1

    # Save stability data for postprocessing:
    if rank==0:
        np.save('./out/data/'+str(DIM)+'d/data_stability.npy', DATA_STABILITY)

    # Version of the code for fixed meshres:
    # prm.meshres=1000
    # global DATA_STABILITY
    # #DATA_STABILITY["meshres"]=prm.meshres
    # lower_bound = -2
    # upper_bound = 2
    # scale=np.arange(lower_bound,upper_bound+1)
    # for k_eps in scale:
    #     em.C_EPS=2.**k_eps
    #     DATA_STABILITY[str(em.C_EPS)]={}
    #     for k_cfl in scale:
    #         global C_CFL
    #         C_CFL=2.**k_cfl
    #         DATA_STABILITY[str(em.C_EPS)][str(C_CFL)]={}
    #         stefan_benchmark()
    # # Save stability data for postprocessing:
    # np.save('./out/data/'+str(DIM)+'d/data_stability.npy', DATA_STABILITY)


# Poznamka z 7.6.: bohuzel nelinearni formulace nekonverguje pro Cinf aproximace heavisida a diraca u cp_eff, pokud zvolis C1/Cinf pro HS/D pak to funguje, pro Cinf/Cinf haze nan pro reziduum, prozkoumej co se deje

# Poznamka z 8.6.: problem je v Cinf aproximaci heaviside funkce
