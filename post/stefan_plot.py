import dolfin
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import sim.params as prm
import post.my_plot as mplt

from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from matplotlib.lines import Line2D
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

DIM=0
R_START=0
R_END=1.

# Graphic parameters for particular FEM solutions:
linestyle={"analytic":{"color":mplt.mypalette[0],
                       "marker":''},
           "EHC":{"color":mplt.mypalette[1],
                  "marker":'o'
           },
           "EHCreg":{"color":mplt.mypalette[2],
                 "marker":'v'
           },
           "TTM":{"color":mplt.mypalette[3],
                  "marker":'s'
           }
}

def graph_temp(dat_timeset,plot_timeset,lambda_,theta_analytic,sim,data_hdf,comm,rank,bbox):

    fig, ax = plt.subplots(1,1)
    #ax.set_xlabel(r"$r\,[\mathrm{m}]$")
    #ax.set_ylabel(r"$\theta\,[\mathrm{K}]$")

    # Pomocne veliciny pro vykreslovani grafu teplotnich poli:
    color_id=1
    names=[r"$t_0$"]
    plots=[]
    xticks=[[prm.R1,2*lambda_*np.sqrt(plot_timeset[0])],[r"$r_1$",r"$s(t_0)$"],]
    
    for t in plot_timeset:
        theta_analytic.t=t
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
        if rank!=0:
            comm.send(y_range,dest=0)
        if rank==0:
            plt.plot(x_range,
                     y_range,
                     label="analytic solution",
                     lw=2.,
                     color=mplt.mypalette[color_id],
                     marker=linestyle["analytic"]["marker"]
            )

        if t in plot_timeset[1:-1]:
            # Create legend with time steps:
            stamp=(t-dat_timeset[0])/(dat_timeset[-1]-dat_timeset[0])
            names.append(r"$t_"+str(color_id-1)+"=t_0+"+str(f"{stamp:.1f}")+"\!\cdot\! \\tau$")
            xticks[0].append(2*lambda_*np.sqrt(t))
            xticks[1].append(r"$s(t_"+str(color_id-1)+")$")

        methodplots=()
            
        for method in sim:
            
            # get the index of t in dat_timeset:
            index,=np.where(np.isclose(dat_timeset,t))

            # Load theta from data_hdf file:
            theta=dolfin.Function(sim[method][1].function_space())
            data_hdf.read(theta,"/theta_"+method+"/vector_%i"%index)
            
            for i,x in enumerate(x_range):
                xp=dolfin.Point(x)
                if bbox.compute_collisions(xp):
                    y_range[i]=theta(xp)
            if rank==0:
                for process in range(comm.size)[1:]:
                    y=comm.recv(source=process)
                    y_range=np.maximum(y_range,y)
            if rank!=0:
                comm.send(y_range,dest=0)
            if rank==0:
                plot,=plt.plot(x_range,
                               y_range,
                               label=method,
                               linestyle='None',
                               color=mplt.mypalette[color_id],
                               marker=linestyle[method]["marker"],
                               markevery=(0.+len(methodplots)*0.1/len(sim),0.1)
                )
                methodplots=methodplots+(plot,)
        if rank==0:
            plots.append(methodplots)
        color_id=color_id+1

    names.append(r"$t_\mathrm{max}$")
    xticks[0].extend([2*lambda_*np.sqrt(plot_timeset[-1]),prm.R2])
    xticks[1].extend([r"$s(t_\mathrm{max})$",r"$r_2$"])
    

    # Create two graph legends, for methods and timesteps:
    if rank==0:
        second_legend_elems = [Line2D([0],[0],color=mplt.mypalette[1],label='analytic')]
        for method in sim:
            leg_element=Line2D([0],[0],color=mplt.mypalette[1],marker=linestyle[method]["marker"],linestyle='None',label=method)
            second_legend_elems.append(leg_element)
        second_legend=plt.legend(handles=second_legend_elems,loc="upper right")
        ax2 = plt.gca().add_artist(second_legend)
        if methodplots:
            ax.legend(plots,names,loc="center right",handler_map={tuple: HandlerTuple(ndivide=len(methodplots))})
        else:
            ax.legend(names,loc="lower left")

        # Make custom ticks:
        ax.set_xticks(xticks[0])
        ax.set_xticklabels(xticks[1])
        ax.set_yticks([prm.theta_m])
        ax.set_yticklabels([r"$\theta_\mathrm{m}$"])
        #------------------------------
        # Save the figure:
        fig.set_size_inches(mplt.set_size(345.,ratio=3*(5**.5-1)/8),forward=True)
        fig.savefig('./out/fig/'+str(DIM)+'d/temp_dist.pdf',
                    format='pdf',
                    bbox_inches='tight',
                    transparent=False
        )
        #--------------------------------------------

def graph_temp_diff(dat_timeset,plot_timeset,theta_analytic,sim,data_hdf,comm,rank,bbox):

    fig, ax = plt.subplots(1,1)

    # Pomocne veliciny pro vykreslovani grafu teplotnich poli:
    color_id=1
    names=[]
    plots=[]
    
    for t in plot_timeset:
        theta_analytic.t=t
        
        # Create legend with time steps:
        stamp=(t-dat_timeset[0])/(dat_timeset[-1]-dat_timeset[0])
        names.append("$t=t_0+"+str(f"{stamp:.1f}")+"\!\cdot\! \\tau$")

        methodplots=()
        
        for method in sim:
            
            # get the index of t in dat_timeset:
            index,=np.where(np.isclose(dat_timeset,t))

            # Load theta from data_hdf file:
            theta=dolfin.Function(sim[method][1].function_space())
            data_hdf.read(theta,"/theta_"+method+"/vector_%i"%index)
            
            for i,x in enumerate(x_range):
                xp=dolfin.Point(x)
                if bbox.compute_collisions(xp):
                    y_range[i]=abs(theta(xp)-theta_analytic(xp))
            if rank==0:
                for process in range(comm.size)[1:]:
                    y=comm.recv(source=process)
                    y_range=np.maximum(y_range,y)
            if rank!=0:
                comm.send(y_range,dest=0)
            if rank==0:
                plot,=plt.plot(x_range,
                               y_range,
                               label=method,
                               linestyle='None',
                               color=mplt.mypalette[color_id],
                               marker=linestyle[method]["marker"],
                               markevery=(0.0+color_id*0.1/len(sim),0.1)
                )
                methodplots=methodplots+(plot,)
        if rank==0:
            plots.append(methodplots)
        color_id=color_id+1

    # Create two graph legends, for methods and timesteps:
    if rank==0:
        first_legend_elements = []
        for method in sim:
            leg_element=Line2D([0],[0],color=mplt.mypalette[1],marker=linestyle[method]["marker"],linestyle='None',label=method)
            first_legend_elements.append(leg_element)
        first_legend=plt.legend(handles=first_legend_elements,loc="upper right")
        ax2 = plt.gca().add_artist(first_legend)
        fig.legend(plots,names,loc="lower left",handler_map={tuple: HandlerTuple(ndivide=3)})
        
        #------------------------------
        # Save the figure:
        fig.set_size_inches(mplt.set_size(345.,ratio=(5**.5-1)/2))
        fig.savefig('./out/fig/'+str(DIM)+'d/temp_dist_diff.pdf',
                    format='pdf',
                    bbox_inches='tight',
                    transparent=False
        )
        #--------------------------------------------

# Graph melting front position:
def graph_front_pos(dat_timeset,lambda_,front_positions,offset=False,ls=False):

    plot_data=[[dat_timeset,2*lambda_*np.sqrt(dat_timeset)]]
    plt.plot(dat_timeset,2*lambda_*np.sqrt(dat_timeset))
    legend=['analytic']

    tau=dat_timeset[-1]-dat_timeset[0]
    timestep=dat_timeset[1]-dat_timeset[0]
    
    for method in front_positions:
        
        if offset:
            uh=np.asarray(front_positions[method])
            u=2*lambda_*np.sqrt(dat_timeset)
            
            # Offset the graph using least-squares method:
            #offset=(-np.inner(uh,u)*sum(uh)+np.inner(uh,uh)*sum(u))/(np.inner(uh,uh)*len(uh)-sum(uh)**2)

            # Offset ze spojiteho vzorce (funguje)
            offset=timestep/tau*sum(u-uh)
        else:
            offset=0

        # Least squares fit:
        if ls:
            uh=np.asarray(front_positions[method])
            a = np.vstack([np.sqrt(dat_timeset),np.ones(len(dat_timeset))]).T
            coeffs=np.dot(np.linalg.inv(np.dot(a.T,a)),np.dot(a.T,uh))
            print(coeffs,(coeffs[0]/2-lambda_)/lambda_)

            # least squares plot:
            plot_data.append([dat_timeset,coeffs[0]*np.sqrt(dat_timeset)+coeffs[1]])
            plt.plot(dat_timeset,coeffs[0]*np.sqrt(dat_timeset)+coeffs[1])
        else:
            # offset plot:
            plot_data.append([dat_timeset,np.asarray(front_positions[method])+offset])
        legend.append(method)
                         
    mplt.graph1d(plot_data,
                 color=mplt.mypalette[:len(front_positions)+1],
                 legend=legend,
                 linestyle=linestyle,
                 axlabels = [r"",
                             r"$s\,[\mathrm{m}]$"],
                 xticks=[[dat_timeset[0],dat_timeset[-1]],[r"$t_0$",r"$t_{\mathrm{max}}$"]],
                 savefig={"width":345./2,"name":'./out/fig/'+str(DIM)+'d/front_pos.pdf'},)

# Graph difference between FEM and analytic melting front position:
def graph_front_pos_diff(timeset,lambda_,front_positions):
    plot_data=[]
    legend=[]
    for method in front_positions:
        plot_data.append([timeset,abs(front_positions[method]-2*lambda_*np.sqrt(timeset))])
        legend.append(method)
    mplt.graph1d(plot_data,
                 color=mplt.mypalette[1:],
                 legend=legend,
                 xticks=[[timeset[0],timeset[-1]],
                         [r"$t_0$",r"$t_{\mathrm{max}}$"]],
                 savefig={"width":345./2,"name":'./out/fig/'+str(DIM)+'d/front_pos_diff.pdf'},
    )

def graph_front_vel(timeset,lambda_,front_positions, interpolation=False, curvefit=False):
    plot_data=[[timeset[1:],lambda_/np.sqrt(timeset[1:])]]
    legend=['analytic']

    tau=timeset[-1]-timeset[0]
    timestep=timeset[1]-timeset[0]

    # Curve fit model:
    def f(x,a,b):
        return a/np.sqrt(x)+b


    # Interpolation
    if interpolation:
        for method in front_positions:
            # spline interpolation
            pos_spline=UnivariateSpline(timeset,front_positions[method],k=3)
            vel_spline=pos_spline.derivative()
        
            plot_data.append([timeset[1:],vel_spline(timeset[1:])])
            legend.append(method)
    # Curve fitting:
    elif curvefit:
        for method in front_positions:
            # curve fit
            c_fit=curve_fit(f,timeset[1:],(np.asarray(front_positions[method][1:])-np.asarray(front_positions[method][:-1]))/timestep)
        
            plot_data.append([timeset,c_fit[0][0]/np.sqrt(timeset)+c_fit[0][1]])
            legend.append(method)
    else:
        for method in front_positions:
    
            plot_data.append([timeset[1:],(np.asarray(front_positions[method][1:])-np.asarray(front_positions[method][:-1]))/timestep])
            legend.append(method)
            
    mplt.graph1d(plot_data,
                 color=2*mplt.mypalette[:len(front_positions)+1],
                 legend=legend,
                 linestyle=linestyle,
                 axlabels = [r"",
                             r"$\mathbf{\nu}_{\sigma}\,[\mathrm{m}/\mathrm{s}]$"],
                 xticks=[[timeset[0],timeset[-1]],[r"$t_0$",r"$t_{\mathrm{max}}$"]],
                 savefig={"width":345./2,
                          "name":'./out/fig/'+str(DIM)+'d/front_vel.pdf'
                 },
    )
