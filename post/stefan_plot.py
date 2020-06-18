import dolfin
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import sim.params as prm
import post.my_plot as mplt

from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from matplotlib.lines import Line2D

DIM=0

# Graphic parameters for particular FEM solutions:
linestyle={"analytic":{"color":mplt.mypalette[0],
                       "marker":''},
           "ehc":{"color":mplt.mypalette[1],
                  "marker":'o'
           },
           "em":{"color":mplt.mypalette[2],
                 "marker":'v'
           },
           "cao":{"color":mplt.mypalette[3],
                  "marker":'s'
           }
}

def graph_temp(dat_timeset,plot_timeset,theta_analytic,sim,data_hdf,comm,rank,bbox):

    fig, ax = plt.subplots(1,1)
    ax.set_xlabel(r"$r\,[\mathrm{m}]$")
    ax.set_ylabel(r"$\theta\,[\mathrm{K}]$")

    # Pomocne veliciny pro vykreslovani grafu teplotnich poli:
    color_id=1
    names=[]
    plots=[]
    first_legend_elems=[]
    
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

    # Create two graph legends, for methods and timesteps:
    if rank==0:
        second_legend_elems = [Line2D([0],[0],color=mplt.mypalette[1],label='analytic')]
        for method in sim:
            leg_element=Line2D([0],[0],color=mplt.mypalette[1],marker=linestyle[method]["marker"],linestyle='None',label=method)
            second_legend_elems.append(leg_element)
        second_legend=plt.legend(handles=second_legend_elems,loc="upper right")
        ax2 = plt.gca().add_artist(second_legend)
        if methodplots:
            ax.legend(plots,names,loc="lower left",handler_map={tuple: HandlerTuple(ndivide=len(methodplots))})
        else:
            ax.legend(names,loc="lower left")
        #------------------------------
        # Save the figure:
        fig.set_size_inches(mplt.set_size(345./2))
        fig.savefig('./out/fig/'+str(DIM)+'d/temp_dist.pdf', format='pdf', bbox_inches='tight', transparent=False)
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
        fig.set_size_inches(mplt.set_size(345./2))
        fig.savefig('./out/fig/'+str(DIM)+'d/temp_dist_diff.pdf', format='pdf', bbox_inches='tight', transparent=False)
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
                 color=mplt.mypalette,
                 legend=legend,
                 axlabels = [r"$t\,[\mathrm{s}]$",
                             r"$s\,[\mathrm{m}]$"],
                 savefig={"width":345./2,"name":'./out/fig/'+str(DIM)+'d/front_pos.pdf'},)

# Graph difference between FEM and analytic melting front position:
def graph_front_pos_diff(dat_timeset,lambda_,front_positions):
    plot_data=[]
    legend=[]
    for method in front_positions:
        plot_data.append([dat_timeset,abs(front_positions[method]-2*lambda_*np.sqrt(dat_timeset))])
        legend.append(method)
    mplt.graph1d(plot_data,color=mplt.mypalette[1:],legend=legend,savefig={"width":345./2,"name":'./out/fig/'+str(DIM)+'d/front_pos_diff.pdf'},)
