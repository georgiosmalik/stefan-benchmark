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
# 1. pri generovani prostoru varianty pro ruzne okrajove podminky: Dirichlet/Neumann
# 2. prostor funkci muzu dostat z objektu DirichletBC, neni tedy treba ve stefan_functionspaces vracet zvlast function space a zvlast boundary_condition
# DONE 3. sloucit hledani lambdy a presne reseni?
# 4. lze ukladat vse do jednoho xdmf souboru?


import dolfin
import numpy as np
import matplotlib.pyplot as plt

import pre.stefan_mesh as msh
import sim.params as prm
import sim.enthalpy_method as em
import post.my_plot as mplt

from scipy.special import erf, erfc, expi, gamma, gammaincc
from scipy.optimize import fsolve

from math import floor, ceil

dolfin.set_log_level(30)

TIMESTEP=0.1
dt=TIMESTEP

def stefan_analytic_sol(dim, ploteq=False):
    """Return analytic solution of the radially symmetric Stefan problem."""

    class MyExpression(dolfin.Expression):
        def update(self, t):
            self.t=t

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
            f = lambda x : prm.rho*prm.L_m*np.sqrt(np.pi)*x + prm.k_l/np.sqrt(prm.kappa_l)*(prm.theta_m - prm.theta_0)/erf(x/np.sqrt(prm.kappa_l))*np.exp(-x*x/prm.kappa_l) + prm.k_s/np.sqrt(prm.kappa_s)*(prm.theta_m - prm.theta_inf)/erfc(x/np.sqrt(prm.kappa_s))*np.exp(-x*x/prm.kappa_s)
            lambda_ = fsolve(f,0.00001)
            if savefig:
                ax, _ = graph_transcendental_eq(lambda_,f,1)
                ax.axhline(y=0, lw=1, color='k')
            return lambda_[0]

        lambda_=transcendental_eq_1d(ploteq)
        code_analytic="""x[0] < 2*lambda_*sqrt(t) ? theta_0 + (theta_m - theta_0)/erf(lambda_/sqrt(kappa_l))*erf(x[0]/(2*sqrt(t*kappa_l))) : theta_inf + (theta_m - theta_inf)/erfc(lambda_/sqrt(kappa_s))*erfc(x[0]/(2*sqrt(t*kappa_s)))"""
        theta_analytic=MyExpression(code_analytic,lambda_=lambda_, t=0.1, theta_m=prm.theta_m, theta_0=prm.theta_0, theta_inf=prm.theta_inf, kappa_l=prm.kappa_l, kappa_s=prm.kappa_s, degree=3)
        code_flux='-k*C1*exp(-r*r/(4*t*kappa))/sqrt(t)'
        q_in=MyExpression(code_flux, t=0.1, k=prm.k_l, C1=(prm.theta_m-prm.theta_0)/(erf(lambda_/np.sqrt(prm.kappa_l))*np.sqrt(np.pi*prm.kappa_l)), r=0, kappa=prm.kappa_l, degree=0)
        q_out=MyExpression(code_flux, t=0.1, k=prm.k_s, C1=(prm.theta_m-prm.theta_inf)/(erfc(lambda_/np.sqrt(prm.kappa_s))*np.sqrt(np.pi*prm.kappa_s)), r=prm.L, kappa=prm.kappa_s, degree=0)
        #q_n = em.Expression('(theta_inf - theta_m)/(erfc(lam/sqrt(kappa_s))*sqrt(pi*kappa_s*t))*exp(-(x[0]*x[0])/(4*t*kappa_s))', t = 0.0, lam = lambda_[0], theta_m = em.theta_m, theta_inf = em.theta_inf, kappa_s = em.kappa_s, domain = mesh, element = V_ele)
        #F = em.k(theta_k)*em.inner(em.grad(_theta), em.grad(theta_))*dx + em.rhoc(theta_k)/dt*(em.inner(_theta, theta_) - em.inner(theta_k, theta_))*dx - em.k(theta_k)*em.inner(q_n,n[0])*theta_*ds(2)
        return lambda_, theta_analytic, q_in, q_out

    def theta_sol_2d():

        def transcendental_eq_2d(savefig):
            f = lambda x : prm.rho*prm.L_m*x**2 - prm.k_l*prm.theta_0*np.exp(-(x**2)/(prm.kappa_l)) + prm.k_s*(prm.theta_m-prm.theta_inf)*np.exp(-(x**2)/prm.kappa_s)/(-expi(-(x**2)/prm.kappa_s))
            lambda_ = fsolve(f,0.00001)
            if savefig:
                ax, _ = graph_transcendental_eq(lambda_,f,2)
                ax.axhline(y=0, lw=1, color='k')
            return lambda_[0]

        lambda_=transcendental_eq_2d(ploteq)
        code_analytic = '''
          #include <math.h>
          #include <boost/math/special_functions/expint.hpp>
          using boost::math::expint;

          namespace dolfin {
               class MyFun : public Expression
              {
                      double t, theta_inf, theta_m, alpha_l, alpha_s, lambda, theta_0;
                      public:
                                MyFun(): Expression() {};
                      void eval(Array<double>& values, const Array<double>& x) const {
                              double f_l = (x[0]*x[0]+x[1]*x[1])/(4*alpha_l*t) ;
                              double f_s = (x[0]*x[0]+x[1]*x[1])/(4*alpha_s*t) ;
                              if ( sqrt(x[0]*x[0]+x[1]*x[1]) <= 2*lambda*sqrt(t) ) {
                                 values[0] = theta_m + theta_0*expint(-lambda*lambda/alpha_l) - theta_0*expint(-f_l);
                              }
                              else {
                                 values[0] = theta_inf - (theta_inf - theta_m)/expint(-lambda*lambda/alpha_s)*expint(-f_s);
                              }
                      }

                      void update_params(double _theta_inf, double _theta_m, double _alpha_l, double _alpha_s, double _lambda, double _theta_0) {
                            theta_inf = _theta_inf;
                            theta_m = _theta_m;
                            alpha_l = _alpha_l;
                            alpha_s = _alpha_s;
                            lambda = _lambda;
                            theta_0 = _theta_0;
                    }
                      void update(double _t) {
                            t = _t;
                      }
              };
        }'''
        theta_analytic=dolfin.Expression(code_analytic, degree=3)
        theta_analytic.update_params(prm.theta_inf, prm.theta_m, prm.kappa_l, prm.kappa_s, lambda_, prm.theta_0)
        theta_analytic.update(0.1)
        code_flux='-k*C1*2*exp(-r*r/(4*kappa*t))/r'
        q_in=MyExpression(code_flux, t=0.1, k=prm.k_l, C1=-prm.theta_0, r=prm.R1, kappa=prm.kappa_l, degree=0)
        q_out=MyExpression(cppcode, t=0.1, k=prm.k_s, C1=(prm.theta_m-prm.theta_inf)/expi(-lambda_**2/prm.kappa_s), r=prm.R2, kappa=prm.kappa_s, degree=0)
        return lambda_, theta_analytic, q_in, q_out

    def theta_sol_3d():

        def transcendental_eq_3d(savefig):
            f = lambda x : prm.rho*prm.L_m*x**3 - prm.k_l*np.sqrt(prm.kappa_l)*prm.theta_0*np.exp(-(x**2)/prm.kappa_l) + prm.k_s*np.sqrt(prm.kappa_s)*(prm.theta_m - prm.theta_inf)*np.exp(-(x**2)/prm.kappa_s)/((-2)*gamma(0.5)*gammaincc(0.5,x**2/prm.kappa_s) + 2*np.sqrt(prm.kappa_s)/x*np.exp(-x**2/prm.kappa_s))
            lambda_ = fsolve(f,0.00001)
            if savefig:
                ax, _ = graph_transcendental_eq(lambda_,f,3)
                ax.axhline(y=0, lw=1, color='k')
            return lambda_[0]

        lambda_=transcendental_eq_3d(ploteq)
        code_analytic = '''
          #include <math.h>
          #include <boost/math/special_functions/gamma.hpp>
          using boost::math::tgamma;

          namespace dolfin {
               class MyFun : public Expression
              {
                      double t, theta_inf, theta_m, alpha_l, alpha_s, lambda, theta_0;
                      public:
                                MyFun(): Expression() {};
                      void eval(Array<double>& values, const Array<double>& x) const {
                              double f_l = (x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/(4*alpha_l*t) ;
                              double f_s = (x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/(4*alpha_s*t) ;
                              if (sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]) <= 2*lambda*sqrt(t)) {
                                 values[0] = theta_0*(-2*tgamma(0.5,f_l) + 2*sqrt(1/f_l)*exp(-f_l)) + theta_m - theta_0*(-2*tgamma(0.5,lambda*lambda/alpha_l) + 2*sqrt(alpha_l)/lambda*exp(-lambda*lambda/alpha_l));
                              }
                              else {
                                 values[0] = theta_inf - (theta_inf - theta_m)/(-2*tgamma(0.5,lambda*lambda/alpha_s) + 2*sqrt(alpha_s)/lambda*exp(-lambda*lambda/alpha_s))*(-2*tgamma(0.5,f_s) + 2*sqrt(1/f_s)*exp(-f_s));
                              }
                      }

                      void update_params(double _theta_inf, double _theta_m, double _alpha_l, double _alpha_s, double _lambda, double _theta_0) {
                            theta_inf = _theta_inf;
                            theta_m = _theta_m;
                            alpha_l = _alpha_l;
                            alpha_s = _alpha_s;
                            lambda = _lambda;
                            theta_0 = _theta_0;
                    }
                      void update(double _t) {
                            t = _t;
                      }
              };
        }'''
        theta_analytic=dolfin.Expression(code, degree=3)
        theta_analytic.update_params(prm.theta_inf, prm.theta_m, prm.kappa_l, prm.kappa_s, lambda_, prm.theta_0)
        theta_analytic.update(0.1)
        code_flux='-k*C1*(-4)*exp(-r*r/(4*kappa*t))*sqrt(kappa*t)/(r*r)'
        q_in=MyExpression(code_flux, t=0.1, k=prm.k_l, C1=prm.theta_0, r=prm.R1, kappa=prm.kappa_l, degree=0)
        q_out=MyExpression(code_flux, t=0.1, k=prm.k_s, C1=(prm.theta_m-prm.theta_inf)/((-2)*gamma(0.5)*gammaincc(0.5,lambda_**2/prm.kappa_s) + 2*np.sqrt(prm.kappa_s)/lambda_*np.exp(-lambda_**2/prm.kappa_s)), r=prm.R2, kappa=prm.kappa_s, degree=0)
        return lamdba_, theta_analytic, q_in, q_out
    
    dimswitch = {
        1:theta_sol_1d,
        2:theta_sol_2d,
        3:theta_sol_3d
        }
    return dimswitch.get(dim, "Please enter 1d, 2d, or 3d.")

def stefan_benchmark_sim(mesh, mesh_analytic, boundary, n, dx, ds, lambda_, theta_analytic, q_in, q_out, methods):

    def stefan_loop_timesets():
        t_start=(0.4/(2*lambda_))**2
        t_end=(0.8/(2*lambda_))**2
        global dt
        dt=int((t_end-t_start)/10000)
        timeset=np.arange(t_start,t_end,dt)
        numplots=10
        plot_timeset=timeset[::int(len(timeset)/(numplots-1))]
        return timeset, plot_timeset

    def stefan_function_spaces(degree=1, nonlinear=False):
        # Tohle by melo vracet definice prostoru a funkce
        T_ele = dolfin.FiniteElement("CG", mesh.ufl_cell(), degree)
        T = dolfin.FunctionSpace(mesh, T_ele)
        boundary_conditions=[dolfin.DirichletBC(T,theta_analytic,boundary[0],boundary[1]['inner']),dolfin.DirichletBC(T,theta_analytic,boundary[0],boundary[1]['outer'])]
        if nonlinear:
            theta = dolfin.Function(T)
            theta_ = dolfin.TestFunction(T)
            _theta = dolfin.Function(T)
            theta_k = dolfin.Function(T)
        else:    
            theta = dolfin.Function(T)
            _theta = dolfin.TrialFunction(T)
            theta_ = dolfin.TestFunction(T)
            theta_k = dolfin.Function(T)
        return (T,boundary_conditions,theta,_theta,theta_,theta_k)

    def stefan_form_update(t):
        theta_analytic.update(t)
        q_in.update(t)
        q_out.update(t)
    
    def stefan_loop():
        # Mel by vratit formulaci v zavislosti na okrajovych podminkach a pouzite metode (equiv heat capacity, cao - temp trans enthalpy method, nase metoda)
        def stefan_problem_form(method):
            # Specifikuje metodu, kterou se bude uloha pocitat
                
            def stefan_form_em():
                # Definuj prostor funkci:
                (T,boundary_conditions,theta,_theta,theta_,theta_k)=stefan_function_spaces()
                # Nastav poc. podminku:
                theta_k=dolfin.project(theta_analytic,T)

                # Nasledujici tri radky specifikuji okrajove podminky problemu, v zavislosti na hodnote i se priradi bud dirichlet-neumann (i=0), N-D (i=1), N-N (i=2), D-D (i=0.5) 
                i=2
                q_form = [q_out*theta_*ds(2),q_in*theta_*ds(1)][floor(-1.5+i):ceil(0.5+i)]
                bcs=boundary_conditions[floor(0+i):ceil(1+i)]
                
                def k_eff(theta):
                    return em.mollify(prm.k_s,prm.k_l,theta,x0=prm.theta_m)
                def c_p_eff(theta):
                    return em.mollify(prm.cp_s,prm.cp_l,theta,x0=prm.theta_m)+em.dirac(prm.cp_m+prm.L_m,theta, x0=prm.theta_m)

                F = k_eff(theta_k)*dolfin.inner(dolfin.grad(_theta), dolfin.grad(theta_))*dx+prm.rho*c_p_eff(theta_k)/dt*(dolfin.inner(_theta, theta_) - dolfin.inner(theta_k, theta_))*dx - sum(q_form)
                problem = dolfin.LinearVariationalProblem(dolfin.lhs(F),dolfin.rhs(F),theta,bcs)
                solver = dolfin.LinearVariationalSolver(problem)
                return solver, theta, theta_k

            def stefan_form_ehc():
                # Definuj prostor funkci:
                (T,boundary_conditions,theta,_theta,theta_,theta_k)=stefan_function_spaces()
                # Nastav poc. podminku:
                theta_k=dolfin.project(theta_analytic,T)

                # Nasledujici tri radky specifikuji okrajove podminky problemu, v zavislosti na hodnote i se priradi bud dirichlet-neumann (i=0), N-D (i=1), N-N (i=2), D-D (i=0.5) 
                i=2
                q_form = [q_out*theta_*ds(2),q_in*theta_*ds(1)][floor(-1.5+i):ceil(0.5+i)]
                bcs=boundary_conditions[floor(0+i):ceil(1+i)]
            
                def k_eff(theta):
                    return em.mollify(prm.k_s,prm.k_l,theta,x0=prm.theta_m)
                def c_p_eff(theta):
                    return em.mollify(prm.cp_s,prm.cp_l,theta,x0=prm.theta_m,deg='disC')+em.dirac(prm.cp_m+prm.L_m,theta,x0=prm.theta_m,deg='disC')

                F = k_eff(theta_k)*dolfin.inner(dolfin.grad(_theta), dolfin.grad(theta_))*dx+prm.rho*c_p_eff(theta_k)/dt*(dolfin.inner(_theta, theta_) - dolfin.inner(theta_k, theta_))*dx - sum(q_form)
                problem = dolfin.LinearVariationalProblem(dolfin.lhs(F),dolfin.rhs(F),theta,bcs)
                solver = dolfin.LinearVariationalSolver(problem)
                return solver, theta, theta_k

            def stefan_form_cao():
                # Definuj prostor funkci:
                (T,boundary_conditions,theta,_theta,theta_,theta_k)=stefan_function_spaces()
                # Nastav poc. podminku:
                theta_k=dolfin.project(theta_analytic,T)

                # Nasledujici tri radky specifikuji okrajove podminky problemu, v zavislosti na hodnote i se priradi bud dirichlet-neumann (i=0), N-D (i=1), N-N (i=2), D-D (i=0.5) 
                i=2
                q_form = [q_out*theta_*ds(2),q_in*theta_*ds(1)][floor(-1.5+i):ceil(0.5+i)]
                bcs=boundary_conditions[floor(0+i):ceil(1+i)]
            
                def k_eff(theta):
                    return em.mollify(prm.k_s,prm.k_l,theta,x0=prm.theta_m,deg='Cinf')
                def c_p_eff(theta):
                    return em.mollify(prm.cp_s,prm.cp_l,theta,x0=prm.theta_m,deg='disC')+em.dirac(prm.cp_m+prm.L_m,theta,x0=prm.theta_m,deg='disC')

                # Cao formulation source term
                def s(theta, theta0=prm.theta_m, eps=em.eps):
                    return dolfin.conditional(abs(theta-theta0)<eps,prm.cp_m*eps + prm.L_m/2, dolfin.conditional(theta>theta0, prm.cp_s*eps + prm.L_m, prm.cp_s*eps))
                # Nonlinear formulation
                # F = k_eff(theta)*dolfin.inner(dolfin.grad(theta), dolfin.grad(theta_))*dx+prm.rho/dt*(c_p_eff(theta)*(theta-dolfin.Constant(prm.theta_m)) + s(theta) - c_p_eff(theta_k)*(theta_k-dolfin.Constant(prm.theta_m))-s(theta_k))*theta_*dx - sum(q_form)
                # problem = dolfin.NonlinearVariationalProblem(F,theta,bcs,dolfin.derivative(F,theta))
                # solver = dolfin.NonlinearVariationalSolver(problem)
                # param = solver.parameters
                # param['newton_solver']['absolute_tolerance'] = 1E-8
                # param['newton_solver']['relative_tolerance'] = 1E-7
                # param['newton_solver']['maximum_iterations'] = 25

                # Linear formulation
                F = k_eff(theta_k)*dolfin.inner(dolfin.grad(_theta), dolfin.grad(theta_))*dx+prm.rho/dt*(c_p_eff(theta_k)*(_theta-dolfin.Constant(prm.theta_m)) + s(theta_k) - c_p_eff(theta_k)*(theta_k-dolfin.Constant(prm.theta_m))-s(theta_k))*theta_*dx - sum(q_form)
                problem = dolfin.LinearVariationalProblem(dolfin.lhs(F),dolfin.rhs(F),theta,bcs)
                solver = dolfin.LinearVariationalSolver(problem)
                
                return solver, theta, theta_k
            
            dimswitch = {
                'em':stefan_form_em,
                'ehc':stefan_form_ehc,
                'cao':stefan_form_cao
            }
            return dimswitch.get(method,"Please enter 'em', 'ehc', or 'cao'.")
            # def stefan_form_1d():
            #     F = k_eff(theta_k)*dolfin.inner(dolfin.grad(_theta), dolfin.grad(theta_))*dx+prm.rho*c_p_eff(theta_k)/dt*(dolfin.inner(_theta, theta_) - dolfin.inner(theta_k, theta_))*dx - q_out*theta_*ds(2)
            #     problem = dolfin.LinearVariationalProblem(dolfin.lhs(F),dolfin.rhs(F),theta,boundary_conditions[0])
            #     solver = dolfin.LinearVariationalSolver(problem)
            #     return solver, theta, theta_k
    
            # def stefan_form_2d():
            #     F = k_eff(theta_k)*dolfin.inner(dolfin.grad(_theta), dolfin.grad(theta_))*dx + prm.rho*c_p_eff(theta_k)/dt*(dolfin.inner(_theta, theta_) - dolfin.inner(theta_k, theta_))*dx - q_in*theta_*ds(1) - q_out*theta_*ds(2)
            #     problem=dolfin.LinearVariationalProblem(dolfin.lhs(F), dolfin.rhs(F),theta)
            #     solver = dolfin.LinearVariationalSolver(problem)
            #     return solver, theta, theta_k
    
            # def stefan_form_3d():
            #     F = k_eff(theta_k)*dolfin.inner(dolfin.grad(_theta), dolfin.grad(theta_))*dx + prm.rho*c_p_eff(theta_k)/dt*(dolfin.inner(_theta, theta_) - dolfin.inner(theta_k, theta_))*dx - q_in*theta_*ds(1) - q_out*theta_*ds(2)
            #     problem=dolfin.LinearVariationalProblem(dolfin.lhs(F), dolfin.rhs(F),theta)
            #     solver = dolfin.LinearVariationalSolver(problem)
            #     return solver, theta, theta_k
    
            # dimswitch = {
            #     1:stefan_form_1d,
            #     2:stefan_form_2d,
            #     3:stefan_form_3d
            # }
            # return dimswitch.get(mesh.geometric_dimension(),"Please enter 1d, 2d, or 3d.")
        
        timeset, plot_timeset=stefan_loop_timesets()
        theta_analytic.update(timeset[0])
        sim={}
        for method in methods:
            (solver, theta, theta_k)=stefan_problem_form(method)()
            sim[method]=(solver,theta,theta_k)
        for t in timeset:
            stefan_form_update(t)
            for method in methods:
                sim[method][0].solve()
                sim[method][2].assign(sim[method][1])
            #stefan_save_data(t,theta,theta_analytic)
            if t in plot_timeset:
                dolfin.plot(theta_analytic, mesh=mesh_analytic, label="analytic solution")
                for method in methods:
                    dolfin.plot(sim[method][1], label=method)
                plt.legend(loc="upper right")
                plt.show()

    return stefan_loop

def stefan1d():
    # preprocessing:
    (mesh,boundary,n,dx,ds)=msh.stefan_mesh(1)(0.,prm.L,prm.nx)
    (mesh_analytic,_,_,_,_)=msh.stefan_mesh(1)(0.,prm.L,3*prm.nx)
    # find lambda:
    (lambda_,theta_analytic,q_in,q_out)=stefan_analytic_sol(1)()
    # do the computation:
    stefan_benchmark_sim(mesh,mesh_analytic,boundary,n,dx,ds,lambda_,theta_analytic,q_in,q_out,['em','ehc','cao'])()

def stefan2d():
    # (mesh,boundary,n,dx,ds) = msh.stefan_mesh('2d')(prm.R1,prm.R2,prm.mshres,stefan=True)
    # lambda_ = stefan_transcendental_eq(2)(savefig=True)
    # (T,theta,_theta,theta_,theta_k) = stefan_functionspaces(mesh)
    # q_R1,q_R2 = stefan_boundary_terms(2,lambda_)
    return None

def stefan3d():
    # (mesh,boundary,n,dx,ds) = msh.stefan_mesh('3d')("./stefan_benchmark/pre/gmsh_mesh/sphere")
    # lambda_ = stefan_transcendental_eq(3)(savefig=True)
    # (T,theta,_theta,theta_,theta_k)=stefan_functionspaces(mesh)
    # q_R1,q_R2 = stefan_boundary_terms(3,lambda_)
    return None

# Hlavni switch, ktery spusti simulaci pro konkretni dimenzi
def stefan_benchmark(dim):
    dimswitch = {
        "1d":stefan1d,
        "2d":stefan2d,
        "3d":stefan3d
        }
    return dimswitch.get(dim,"Please enter 1d, 2d, or 3d.")
