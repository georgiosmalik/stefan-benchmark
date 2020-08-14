import os
import sys
import dolfin
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import sim.params as prm
import post.my_plot as mplt

from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import UnivariateSpline, LinearNDInterpolator
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

DIM=0

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

def load_data():
    
    global DATA_PY

    try:
        DATA_PY=np.load('./out/data/'+str(DIM)+'d/data.npy',allow_pickle='TRUE').item()
    except FileNotFoundError:
        print("Benchmark data file N/A, run benchmark first.")

def load_data_stability():
    
    global DATA_STABILITY

    try:
        DATA_STABILITY = np.load('./out/data/'+str(DIM)+'d/data_stability.npy',allow_pickle='TRUE').item()
    except FileNotFoundError:
        print("Stability data file N/A, run stability first.")

    return DATA_STABILITY

def graph_temp():

    data=DATA_PY

    r1=data["problem_params"]["r1"]
    r2=data["problem_params"]["r2"]
    lambda_=data["problem_params"]["lambda"]
    timeset=list(map(float,list(data["temp_dist"].keys())))
    methods=list(data["temp_dist"][str(timeset[0])].keys())[1:]

    fig, ax = plt.subplots(1,1)

    # Pomocne veliciny pro vykreslovani grafu teplotnich poli:
    color_id=1
    names=[r"$t_0$"]
    plots=[]

    xticks=[[r1,2*lambda_*np.sqrt(timeset[0])],[r"$r_1$",r"$s(t_0)$"],]
    
    for t in timeset:
        x_range=data["temp_dist"][str(t)]["analytic"][0]
        y_range=data["temp_dist"][str(t)]["analytic"][1]

        plt.plot(x_range,
                 y_range,
                 lw=2.,
                 color=mplt.mypalette[color_id],
                 marker=linestyle["analytic"]["marker"]
        )

        if t in timeset[1:-1]:
            # Create legend with time steps:
            stamp=(t-timeset[0])/(timeset[-1]-timeset[0])
            names.append(r"$t_"+str(color_id-1)+"{=}t_0{+}"+str(f"{stamp:.1f}")+"\!\cdot\! \\tau$")
            xticks[0].append(2*lambda_*np.sqrt(t))
            xticks[1].append(r"$s(t_"+str(color_id-1)+")$")

        methodplots=()
            
        for method in methods:

            x_range=data["temp_dist"][str(t)][method][0]
            y_range=data["temp_dist"][str(t)][method][1]

            plot,=plt.plot(x_range,
                           y_range,
                           label=method,
                           linestyle='None',
                           color=mplt.mypalette[color_id],
                           marker=linestyle[method]["marker"],
                           markersize=4,
                           markevery=(0.+len(methodplots)*0.1/len(methods),0.1)
            )
            methodplots=methodplots+(plot,)
            
        plots.append(methodplots)
        color_id=color_id+1

    names.append(r"$t_\mathrm{max}$")
    xticks[0].extend([2*lambda_*np.sqrt(timeset[-1]),r2])
    xticks[1].extend([r"$s(t_\mathrm{max})$",r"$r_2$"])
    

    # Create two graph legends, for methods and timesteps:
    second_legend_elems = [Line2D([0],[0],color=mplt.mypalette[1],label='analytic')]
    for method in methods:
        leg_element=Line2D([0],[0],color=mplt.mypalette[1],marker=linestyle[method]["marker"],markersize=4,linestyle='None',label=method)
        second_legend_elems.append(leg_element)
    second_legend=plt.legend(handles=second_legend_elems,loc="upper right", frameon=True, fancybox=False, borderaxespad=0.)
    ax2 = plt.gca().add_artist(second_legend)
    if methodplots:
        ax.legend(plots,names,loc="upper right", bbox_to_anchor=(0.685,1.),handler_map={tuple: HandlerTuple(ndivide=len(methodplots))}, frameon=True, fancybox=False, borderaxespad=0.)
    else:
        ax.legend(names,loc="upper right", bbox_to_anchor=(0.685,1.), frameon=True, fancybox=False, borderaxespad=0.)

    # Make custom ticks:
    ax.set_xticks(xticks[0])
    ax.set_xticklabels(xticks[1])
    ax.set_yticks([prm.theta_m])
    ax.set_yticklabels([r"$\theta_\mathrm{m}$"])

    #------------------------------
    # Save the figure:
    h_max=data["disc_params"]["h_max"]
    
    eps=data["disc_params"]["eps"]
    h_eps=data["disc_params"]["h_eps"]
    c_eps=data["disc_params"]["C_eps"]
    
    dt=data["disc_params"]["dt"]
    c_cfl=data["disc_params"]["C_CFL"]

    fig.set_size_inches(mplt.set_size(345./2),forward=True)
    fig.savefig('./out/fig/'+str(DIM)+'d/temp_dist_(eps='+'{0:>1.1e}'.format(eps)+'(h_eps='+'{0:>1.2e}'.format(h_eps)+',C_eps='+'{0:>1.1e}'.format(c_eps)+'),h_max='+'{0:>1.2e}'.format(h_max)+',dt='+'{0:>1.2e}'.format(dt)+'(C_CFL='+'{0:>1.1e}'.format(c_cfl)+').pdf',
                format='pdf',
                bbox_inches='tight',
                transparent=False
    )
    #--------------------------------------------

# Graph melting front position:
def graph_front_pos(offset=False,ls=False):

    data=DATA_PY

    lambda_=data["problem_params"]["lambda"]
    timeset=data["problem_params"]["sim_timeset"]
    methods=list(data["front_pos"].keys())

    plot_data=[[timeset,2*lambda_*np.sqrt(timeset)]]
    legend=['analytic']

    tau=timeset[-1]-timeset[0]

    # We compute with a fixed timestep
    timestep=timeset[1]-timeset[0]
    
    for method in methods:
        front_positions=data["front_pos"][method]
        if offset:
            uh=np.asarray(front_positions)
            u=2*lambda_*np.sqrt(timeset)
            
            # Offset the graph using least-squares method:
            #offset=(-np.inner(uh,u)*sum(uh)+np.inner(uh,uh)*sum(u))/(np.inner(uh,uh)*len(uh)-sum(uh)**2)

            # Offset ze spojiteho vzorce (funguje)
            offset=timestep/tau*sum(u-uh)
        else:
            offset=0

        # Least squares fit:
        if ls:
            uh=np.asarray(front_positions)
            a = np.vstack([np.sqrt(timeset),np.ones(len(timeset))]).T
            coeffs=np.dot(np.linalg.inv(np.dot(a.T,a)),np.dot(a.T,uh))
            print(coeffs,(coeffs[0]/2-lambda_)/lambda_)

            # least squares plot:
            plot_data.append([timeset,coeffs[0]*np.sqrt(timeset)+coeffs[1]])
            plt.plot(timeset,coeffs[0]*np.sqrt(timeset)+coeffs[1])
        else:
            # offset plot:
            plot_data.append([timeset,np.asarray(front_positions)+offset])
        legend.append(method)

    # Save the figure:
    h_max=data["disc_params"]["h_max"]
    
    eps=data["disc_params"]["eps"]
    h_eps=data["disc_params"]["h_eps"]
    c_eps=data["disc_params"]["C_eps"]
    
    dt=data["disc_params"]["dt"]
    c_cfl=data["disc_params"]["C_CFL"]
                         
    mplt.graph1d(plot_data,
                 color=mplt.mypalette[:len(methods)+1],
                 legend=legend,
                 linestyle=linestyle,
                 axlabels=["",r"$s(t)$"],
                 xticks=[[timeset[0],timeset[-1]],[r"$t_0$",r"$t_{\mathrm{max}}$"]],
                 yticks=[[],[]],
                 ylim={"bottom":0.},
                 savefig={"width":345./2,"name":'./out/fig/'+str(DIM)+'d/front_pos_(eps='+'{0:>1.1e}'.format(eps)+'(h_eps='+'{0:>1.2e}'.format(h_eps)+',C_eps='+'{0:>1.1e}'.format(c_eps)+'),h_max='+'{0:>1.2e}'.format(h_max)+',dt='+'{0:>1.2e}'.format(dt)+'(C_CFL='+'{0:>1.1e}'.format(c_cfl)+').pdf'},)

def graph_front_vel():

    data=DATA_PY

    # Load parameters of computation
    lambda_=data["problem_params"]["lambda"]

    h_max=data["disc_params"]["h_max"]
    
    eps=data["disc_params"]["eps"]
    h_eps=data["disc_params"]["h_eps"]
    c_eps=data["disc_params"]["C_eps"]

    timeset=data["problem_params"]["sim_timeset"]
    dt=data["disc_params"]["dt"]
    c_cfl=data["disc_params"]["C_CFL"]
    
    tau=timeset[-1]-timeset[0]
    
    methods=list(data["front_pos"].keys())
    
    plot_data=[[timeset,lambda_/np.sqrt(timeset)]]
    legend=['analytic']

    for method in methods:
        
        front_positions=data["front_pos"][method]

        spline_every = 50
        pos_spline = CubicSpline(timeset[::spline_every], front_positions[::spline_every])
        
        vel_spline = pos_spline(timeset, 1)
        vel_spline = pos_spline.derivative()
        
        plot_data.append([timeset,vel_spline(timeset)])
        legend.append(method)
            
    # Save the figure:
    mplt.graph1d(plot_data,
                 color=2*mplt.mypalette[:len(methods)+1],
                 legend=legend,
                 linestyle=linestyle,
                 axlabels=["",r"$\mathbf{\nu}_{\sigma}(t)$"],
                 xticks=[[timeset[0],timeset[-1]],[r"$t_0$",r"$t_{\mathrm{max}}$"]],
                 yticks=[[],[]],
                 ylim={"bottom":0.},
                 savefig={"width":345./2,
                          "name":'./out/fig/'+str(DIM)+'d/front_vel_(eps='+'{0:>1.1e}'.format(eps)+'(h_eps='+'{0:>1.2e}'.format(h_eps)+',C_eps='+'{0:>1.1e}'.format(c_eps)+'),h_max='+'{0:>1.2e}'.format(h_max)+',dt='+'{0:>1.2e}'.format(dt)+'(C_CFL='+'{0:>1.1e}'.format(c_cfl)+').pdf'
                 },
    )

def graph_stability1p():

    data=DATA_STABILITY['1p']

    disc_params = data.pop('disc_params', None)

    h = disc_params['h']
    c_cfl = disc_params['C_CFL']

    for method in data:

        X = list(map(float,list(data[method].keys())))

        X_C_eps = []
        
        fp_err = []
        l2_err = []
        linf_err = []

        for eps in X:

            X_C_eps.append(data[method][eps]["C_eps"])
            
            fp_err.append(data[method][eps]["fp_err"])
            l2_err.append(data[method][eps]["l2_err"])
            linf_err.append(data[method][eps]["linf_err"])

        fig = plt.figure()
        ax = fig.add_subplot(111, label = "1")
        ax2 = ax.twiny()
        
        ax.plot(X, fp_err, color = mplt.mypalette[0], label = r'$|s-\overline{s}|/|s|$')
        ax.plot(X, l2_err, color = mplt.mypalette[2], linestyle = 'dotted', lw = 2, label = r'$\|\theta-\overline{\theta}\|_{L^2}/\|\theta\|_{L^2}$')
        #ax.plot(X,linf_err, color = mplt.mypalette[2], linestyle = 'dotted', lw = 2., label = r'$\|\theta-\overline{\theta}\|_{L^\infty}/\|\theta\|_{L^\infty}$')
        ax.set_xlabel(r"$\epsilon\,[K]$")
        ax.invert_xaxis()

        ax2.set_xlabel(r"$C_\epsilon\,[-]$")
        ax2.set_xlim(min(X_C_eps), max(X_C_eps))

        ax.set_yscale('log')

        ax.legend(frameon=True, fancybox=False, borderaxespad=0.)
        
        fig.set_size_inches(mplt.set_size(345/2.,ratio=1),forward=True)
        
        fig.savefig('./out/fig/'+str(DIM)+'d/stability1p_'+method+'(h='+'{0:>1.2e}'.format(h)+',C_CFL='+'{0:>1.2e}'.format(c_cfl)+').pdf',
                    format='pdf',
                    bbox_inches='tight',
                    transparent=True
        )
        
def graph_stability2p():

    data=DATA_STABILITY['2p']

    disc_params = data.pop('disc_params', None)

    eps = disc_params['eps']
    h_eps = disc_params['h_eps']
    dt_cfl = disc_params['dt_cfl']

    # Set colormap for stability graph:
    def cmp_(k,method):
        vals = np.ones((k*128+1, 4))
        vals[:k*128, 0] = np.linspace((2-k)*128/256, 1, k*128)
        vals[:k*128, 1] = np.linspace((2-k)*128/256, 1, k*128)
        vals[:k*128, 2] = np.linspace((2-k)*128/256, 1, k*128)
        vals[-1,0:3] = linestyle[method]["color"]
        vals[-1,0:3] = mplt.mypalette[0]
        return ListedColormap(vals)

    mycmp = {"TTM":cmp_(1,"TTM"),"EHC":cmp_(1,"EHC")}

    meshres = list(map(float,list(data['EHC'].keys())))
    timestep = list(map(float,list(data['EHC'][list(data['EHC'].keys())[0]].keys())))

    # zlabel dictionary
    zlabel = {'fp_err':r'$|s-\overline{s}|/s$',
              'l2_err':r'$\|\theta-\overline{\theta}\|_{L^2}/\|\theta\|_{L^2}$',
              'linf_err':r'$\|\theta-\overline{\theta}\|_{L^\infty}/\|\theta\|_{L^\infty}$'
    }

    # Optimality bounds:
    h_opt = np.log10(h_eps)
    deltat_opt = np.log10(dt_cfl)

    # Set x-y ticks:
    X=np.log10(meshres)
    Y=np.log10(timestep)
    
    def log_label(e):
        return r"$10^{"+str(int(e))+"}$"
    
    xticks=np.fromiter((x for x in X if (int(x)==x and abs(x-h_opt)>0.5)), dtype=X.dtype)
    xticks_labels=list(map(log_label,xticks))

    xticks = np.append(xticks,h_opt)
    xticks_labels.append(r"$C_\epsilon {\approx} 1$")

    yticks=np.fromiter((y for y in Y if int(y)==y and abs(y-deltat_opt)>0.5), dtype=Y.dtype)
    yticks_labels=list(map(log_label,yticks))

    yticks = np.append(yticks,deltat_opt)
    c_cfl_label = r"$C_{\mathrm{CFL}} {\approx} 1$"
    yticks_labels.append(c_cfl_label)

    methods=list(data[list(data.keys())[0]][list(data[list(data.keys())[0]].keys())[0]].keys())

    X_gr, Y_gr = np.meshgrid(X,Y)
    positions = np.vstack([X_gr.ravel(),Y_gr.ravel()])
    Z = {}
    Z_log={}

    # contour plot (lines starts with ##)
    ##mask = {}

    for err_type in ['fp_err','l2_err','linf_err']:
        for method in data:
            Z=[]
            Z_log=[]
            ##mask[method]=[]
            for deltat in timestep:
                line=[]
                ##mask_line = []
                for h in meshres:
                    line.append(data[method][h][deltat][err_type])
                    ##if data[x][y][method] == 1:
                    ##mask_line.append(True)
                ##else:
                    ##mask_line.append(False)
                Z.append(line)
            ##mask[method].append(mask_line)
            Z=np.asarray(Z)
            ##mask[method]=np.asarray(mask[method],dtype=bool)

            Z_log=np.log10(Z)
            ##Z_log[method]=np.ma.array(Z_log[method],mask=mask[method])

            # Contour plot
            ##fig, ax = plt.subplots(1,1)

            # 3d plot:
            fig = plt.figure()

            # Add right padding for zlabel to show properly:
            fig.subplots_adjust(right = 0.8)
            
            ax = fig.gca(projection='3d')
            surf = ax.plot_surface(X_gr,Y_gr,Z_log, linewidth=0.5, alpha=0.9, cmap=mycmp[method])
        
            # Contour plot:
            # v = {"TTM":np.linspace(-2.5,0.,11, endpoint=True), "EHC":np.linspace(-5.,0.,21, endpoint=True)}
        
            # cs = ax.contourf(X,Y,Z_log[method], v[method], cmap=mycmp[method], corner_mask=True)
            # ax.contour(cs, linestyles='solid', linewidths=0.5, colors='k')

            # ax.plot(np.ma.array(X,mask=~mask[method]),Y, linestyle='none', color=mplt.mypalette[0], marker='s', markersize=2,label='not converged')
            # plt.legend(loc="lower right", bbox_to_anchor=(1.,0.), frameon=True, fancybox=False, borderaxespad=0.)
            # plt.grid(b=None)
            
            # cbar = fig.colorbar(cs, ticks=v[method])
            # cbar.ax.set_ylabel(r"$\Delta s_{\mathrm{r}}$")
            #------------------
            
            ax.set_aspect('equal','box')
            
            ax.set_xlabel(r"$\Delta_x\,[m]$", labelpad = 0.)
            ax.invert_xaxis()
            
            ax.set_ylabel(r"$\Delta_t\,[s]$", labelpad = 0.)
            
            ax.set_zlabel(zlabel[err_type], labelpad = 0., rotation = 'vertical')
            
            color = mplt.mypalette[0]
            
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks_labels)
        
            ax.xaxis.get_ticklabels()[-1].set_color(color)
            ax.xaxis.get_ticklabels()[-1].set_fontsize(5)
            #ax.xaxis.get_ticklabels()[-1].set_rotation(-60)
            ax.xaxis.get_ticklines()[-1].set_color(color)
            
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks_labels)
            
            ax.yaxis.get_ticklabels()[-1].set_color(color)
            ax.yaxis.get_ticklabels()[-1].set_fontsize(5)
            #ax.yaxis.get_ticklabels()[-1].set_rotation(30)
            ax.yaxis.get_ticklines()[-1].set_color(color)

            ax.tick_params(axis = 'x', pad = -1.5)
            ax.tick_params(axis = 'y', pad = -1.5)
            ax.tick_params(axis = 'z', pad = 0.5)

            for xticklabel in ax.get_xmajorticklabels():
                if xticklabel.get_text() != r"$C_\epsilon {\approx} 1$":
                    xticklabel.set_horizontalalignment('right')
                else:
                    xticklabel.set_horizontalalignment('center')
                    xticklabel.set_verticalalignment('bottom')

            for yticklabel in ax.get_ymajorticklabels():
                if yticklabel.get_text() != c_cfl_label:
                    yticklabel.set_horizontalalignment('left')
                else:
                    yticklabel.set_horizontalalignment('center')
                    yticklabel.set_verticalalignment('bottom')
            
            zticks=np.linspace(int(ax.get_zlim()[0]),0,abs(int(ax.get_zlim()[0]))+1)
            zticks_labels=list(map(log_label,zticks[:-1]))
            zticks_labels.append(r"$1$")
            
            ax.set_zticks(zticks)
            ax.set_zticklabels(zticks_labels)
            
            ##ax.yaxis.get_ticklabels()[-1].set_rotation(90)
            ##ax.yaxis.get_ticklabels()[-1].set_verticalalignment('center')
            

            # Optimality boundary:
            
            # Plot optimality boundary in x-y plane:
            h_opt_line = [h_opt*np.ones(11),np.linspace(ax.get_ylim()[0],deltat_opt,11)]
            deltat_opt_line = [np.linspace(ax.get_xlim()[1],h_opt,11),deltat_opt*np.ones(11)]
            
            ax.plot(*h_opt_line, zs=ax.get_zlim()[0], zdir='z', linestyle = 'dotted', lw = 1, color=color)
            ax.plot(*deltat_opt_line, zs=ax.get_zlim()[0], zdir='z', linestyle = 'dotted', lw = 1, color=color)
            
            # Project optimality boundary on the surface of graph:
            surf_int = LinearNDInterpolator(np.transpose(positions),np.vstack(Z_log.ravel()))
            ax.plot(*h_opt_line,np.transpose(surf_int(list(zip(*h_opt_line))))[0], lw = 1, color=color)
            ax.plot(*deltat_opt_line,np.transpose(surf_int(list(zip(*deltat_opt_line))))[0], lw = 1, color=color)
            
            ax.view_init(elev=20., azim=-35)
            
            fig.set_size_inches(mplt.set_size(345/2.,ratio=1),forward=True)
            
            fig.savefig('./out/fig/'+str(DIM)+'d/stability2p_'+method+'_'+err_type+'(eps='+'{0:>1.2e}'.format(eps)+').pdf',
                        format='pdf',
                        #bbox_inches='tight',
                        transparent=True
        )
    
#----------------------------------
# Backup (HDF file postprocessing):
#==================================

# HDF temp. dist. graphing:

# def graph_temp(dat_timeset,plot_timeset,lambda_,theta_analytic,sim,data_hdf,comm,rank,bbox):

#     fig, ax = plt.subplots(1,1)
#     #ax.set_xlabel(r"$r\,[\mathrm{m}]$")
#     #ax.set_ylabel(r"$\theta\,[\mathrm{K}]$")

#     # Pomocne veliciny pro vykreslovani grafu teplotnich poli:
#     color_id=1
#     names=[r"$t_0$"]
#     plots=[]
#     xticks=[[prm.R1,2*lambda_*np.sqrt(plot_timeset[0])],[r"$r_1$",r"$s(t_0)$"],]
    
#     for t in plot_timeset:
#         theta_analytic.t=t
#         x_range = np.arange(prm.R1,prm.R2,0.001)
#         y_range = x_range*0.0
#         for i,x in enumerate(x_range):
#             xp=dolfin.Point(x)
#             if bbox.compute_collisions(xp):
#                 y_range[i]=theta_analytic(xp)
#         if rank==0:
#             for process in range(comm.size)[1:]:
#                 y=comm.recv(source=process)
#                 y_range=np.maximum(y_range,y)
#         if rank!=0:
#             comm.send(y_range,dest=0)
#         if rank==0:
#             plt.plot(x_range,
#                      y_range,
#                      lw=2.,
#                      color=mplt.mypalette[color_id],
#                      marker=linestyle["analytic"]["marker"]
#             )

#         if t in plot_timeset[1:-1]:
#             # Create legend with time steps:
#             stamp=(t-dat_timeset[0])/(dat_timeset[-1]-dat_timeset[0])
#             names.append(r"$t_"+str(color_id-1)+"{=}t_0{+}"+str(f"{stamp:.1f}")+"\!\cdot\! \\tau$")
#             xticks[0].append(2*lambda_*np.sqrt(t))
#             xticks[1].append(r"$s(t_"+str(color_id-1)+")$")

#         methodplots=()
            
#         for method in sim:
            
#             # get the index of t in dat_timeset:
#             index,=np.where(np.isclose(dat_timeset,t))

#             # Load theta from data_hdf file:
#             theta=dolfin.Function(sim[method][1].function_space())
#             data_hdf.read(theta,"/theta_"+method+"/vector_%i"%index)
            
#             for i,x in enumerate(x_range):
#                 xp=dolfin.Point(x)
#                 if bbox.compute_collisions(xp):
#                     y_range[i]=theta(xp)
#             if rank==0:
#                 for process in range(comm.size)[1:]:
#                     y=comm.recv(source=process)
#                     y_range=np.maximum(y_range,y)
#             if rank!=0:
#                 comm.send(y_range,dest=0)
#             if rank==0:
#                 plot,=plt.plot(x_range,
#                                y_range,
#                                label=method,
#                                linestyle='None',
#                                color=mplt.mypalette[color_id],
#                                marker=linestyle[method]["marker"],
#                                markersize=6,
#                                markevery=(0.+len(methodplots)*0.1/len(sim),0.1)
#                 )
#                 methodplots=methodplots+(plot,)
#         if rank==0:
#             plots.append(methodplots)
#         color_id=color_id+1

#     names.append(r"$t_\mathrm{max}$")
#     xticks[0].extend([2*lambda_*np.sqrt(plot_timeset[-1]),prm.R2])
#     xticks[1].extend([r"$s(t_\mathrm{max})$",r"$r_2$"])
    

#     # Create two graph legends, for methods and timesteps:
#     if rank==0:
#         second_legend_elems = [Line2D([0],[0],color=mplt.mypalette[1],label='analytic')]
#         for method in sim:
#             leg_element=Line2D([0],[0],color=mplt.mypalette[1],marker=linestyle[method]["marker"],linestyle='None',label=method)
#             second_legend_elems.append(leg_element)
#         second_legend=plt.legend(handles=second_legend_elems,loc="upper right", frameon=True, fancybox=False, borderaxespad=0.)
#         ax2 = plt.gca().add_artist(second_legend)
#         if methodplots:
#             ax.legend(plots,names,loc="upper right", bbox_to_anchor=(0.78,1.),handler_map={tuple: HandlerTuple(ndivide=len(methodplots))}, frameon=True, fancybox=False, borderaxespad=0.)
#         else:
#             ax.legend(names,loc="upper right", bbox_to_anchor=(0.78,1.), frameon=True, fancybox=False, borderaxespad=0.)

#         # Make custom ticks:
#         ax.set_xticks(xticks[0])
#         ax.set_xticklabels(xticks[1])
#         ax.set_yticks([prm.theta_m])
#         ax.set_yticklabels([r"$\theta_\mathrm{m}$"])
#         #------------------------------
#         # Save the figure:
#         data_hdf.read(theta,"C_eps")
#         c_eps=theta.vector().norm('linf')

#         data_hdf.read(theta,"C_CFL")
#         c_cfl=theta.vector().norm('linf')

#         data_hdf.read(theta,"h_max")
#         h_max=theta.vector().norm('linf')
        
#         fig.set_size_inches(mplt.set_size(345.,ratio=3*(5**.5-1)/8),forward=True)
#         fig.savefig('./out/fig/'+str(DIM)+'d/temp_dist_(h_max='+'{0:>2.2e}'.format(h_max)+',C_eps='+'{0:>2.1e}'.format(c_eps)+'C_CFL='+'{0:>2.1e}'.format(c_cfl)+').pdf',
#                     format='pdf',
#                     bbox_inches='tight',
#                     transparent=False
#         )
#----------------------------------

# HDF Graph temp. dist. diff.:

# def graph_temp_diff(dat_timeset,plot_timeset,theta_analytic,sim,data_hdf,comm,rank,bbox):

#     fig, ax = plt.subplots(1,1)

#     # Pomocne veliciny pro vykreslovani grafu teplotnich poli:
#     color_id=1
#     names=[]
#     plots=[]
    
#     for t in plot_timeset:
#         theta_analytic.t=t
        
#         # Create legend with time steps:
#         stamp=(t-dat_timeset[0])/(dat_timeset[-1]-dat_timeset[0])
#         names.append("$t{=}t_0{+}"+str(f"{stamp:.1f}")+"\!\cdot\! \\tau$")

#         methodplots=()
        
#         for method in sim:
            
#             # get the index of t in dat_timeset:
#             index,=np.where(np.isclose(dat_timeset,t))

#             # Load theta from data_hdf file:
#             theta=dolfin.Function(sim[method][1].function_space())
#             data_hdf.read(theta,"/theta_"+method+"/vector_%i"%index)
            
#             for i,x in enumerate(x_range):
#                 xp=dolfin.Point(x)
#                 if bbox.compute_collisions(xp):
#                     y_range[i]=abs(theta(xp)-theta_analytic(xp))
#             if rank==0:
#                 for process in range(comm.size)[1:]:
#                     y=comm.recv(source=process)
#                     y_range=np.maximum(y_range,y)
#             if rank!=0:
#                 comm.send(y_range,dest=0)
#             if rank==0:
#                 plot,=plt.plot(x_range,
#                                y_range,
#                                label=method,
#                                linestyle='None',
#                                color=mplt.mypalette[color_id],
#                                marker=linestyle[method]["marker"],
#                                markevery=(0.0+color_id*0.1/len(sim),0.1)
#                 )
#                 methodplots=methodplots+(plot,)
#         if rank==0:
#             plots.append(methodplots)
#         color_id=color_id+1

#     # Create two graph legends, for methods and timesteps:
#     if rank==0:
#         first_legend_elements = []
#         for method in sim:
#             leg_element=Line2D([0],[0],color=mplt.mypalette[1],marker=linestyle[method]["marker"],linestyle='None',label=method)
#             first_legend_elements.append(leg_element)
#         first_legend=plt.legend(handles=first_legend_elements,loc="upper right")
#         ax2 = plt.gca().add_artist(first_legend)
#         fig.legend(plots,names,loc="lower left",handler_map={tuple: HandlerTuple(ndivide=3)})
        
#         #------------------------------
#         # Save the figure:
#         fig.set_size_inches(mplt.set_size(345.,ratio=(5**.5-1)/2))
#         fig.savefig('./out/fig/'+str(DIM)+'d/temp_dist_diff.pdf',
#                     format='pdf',
#                     bbox_inches='tight',
#                     transparent=False
#         )
# #--------------------------------------------

# Graph front positions:

# def graph_front_pos(dat_timeset,lambda_,front_positions,offset=False,ls=False):

#     plot_data=[[dat_timeset,2*lambda_*np.sqrt(dat_timeset)]]
#     plt.plot(dat_timeset,2*lambda_*np.sqrt(dat_timeset))
#     legend=['analytic']

#     tau=dat_timeset[-1]-dat_timeset[0]
#     timestep=dat_timeset[1]-dat_timeset[0]
    
#     for method in front_positions:
        
#         if offset:
#             uh=np.asarray(front_positions[method])
#             u=2*lambda_*np.sqrt(dat_timeset)
            
#             # Offset the graph using least-squares method:
#             #offset=(-np.inner(uh,u)*sum(uh)+np.inner(uh,uh)*sum(u))/(np.inner(uh,uh)*len(uh)-sum(uh)**2)

#             # Offset ze spojiteho vzorce (funguje)
#             offset=timestep/tau*sum(u-uh)
#         else:
#             offset=0

#         # Least squares fit:
#         if ls:
#             uh=np.asarray(front_positions[method])
#             a = np.vstack([np.sqrt(dat_timeset),np.ones(len(dat_timeset))]).T
#             coeffs=np.dot(np.linalg.inv(np.dot(a.T,a)),np.dot(a.T,uh))
#             print(coeffs,(coeffs[0]/2-lambda_)/lambda_)

#             # least squares plot:
#             plot_data.append([dat_timeset,coeffs[0]*np.sqrt(dat_timeset)+coeffs[1]])
#             plt.plot(dat_timeset,coeffs[0]*np.sqrt(dat_timeset)+coeffs[1])
#         else:
#             # offset plot:
#             plot_data.append([dat_timeset,np.asarray(front_positions[method])+offset])
#         legend.append(method)
                         
#     mplt.graph1d(plot_data,
#                  color=mplt.mypalette[:len(front_positions)+1],
#                  legend=legend,
#                  linestyle=linestyle,
#                  axlabels=["",r"$s(t)$"],
#                  xticks=[[dat_timeset[0],dat_timeset[-1]],[r"$t_0$",r"$t_{\mathrm{max}}$"]],
#                  #yticks=[[2*lambda_*np.sqrt(dat_timeset[0]),2*lambda_*np.sqrt(dat_timeset[-1])],[r"$s(t_0)$",r"$s(t_{\mathrm{max}})$"]],
#                  yticks=[[],[]],
#                  savefig={"width":345./2,"name":'./out/fig/'+str(DIM)+'d/front_pos.pdf'},)

# Graph front pos diff (does NOT work for data_py):

# def graph_front_pos_diff(timeset,lambda_,front_positions):
#     plot_data=[]
#     legend=[]
#     for method in front_positions:
#         plot_data.append([timeset,abs(front_positions[method]-2*lambda_*np.sqrt(timeset))])
#         legend.append(method)
#     mplt.graph1d(plot_data,
#                  color=mplt.mypalette[1:],
#                  legend=legend,
#                  xticks=[[timeset[0],timeset[-1]],
#                          [r"$t_0$",r"$t_{\mathrm{max}}$"]],
#                  savefig={"width":345./2,"name":'./out/fig/'+str(DIM)+'d/front_pos_diff.pdf'},
#     )

# Graph front pos vel (does NOT work for data_py):

# def graph_front_vel(timeset,lambda_,front_positions, interpolation=True, curvefit=False):
#     plot_data=[[timeset[1:],lambda_/np.sqrt(timeset[1:])]]
#     legend=['analytic']

#     tau=timeset[-1]-timeset[0]
#     timestep=timeset[1]-timeset[0]

#     # Curve fit model:
#     def f(x,a,b):
#         return a/np.sqrt(x)+b


#     # Interpolation
#     if interpolation:
#         for method in front_positions:
#             # spline interpolation
#             pos_spline=UnivariateSpline(timeset,front_positions[method],k=3)
#             vel_spline=pos_spline.derivative()
        
#             plot_data.append([timeset[1:],vel_spline(timeset[1:])])
#             legend.append(method)
#     # Curve fitting:
#     elif curvefit:
#         for method in front_positions:
#             # curve fit
#             c_fit=curve_fit(f,timeset[1:],(np.asarray(front_positions[method][1:])-np.asarray(front_positions[method][:-1]))/timestep)
        
#             plot_data.append([timeset,c_fit[0][0]/np.sqrt(timeset)+c_fit[0][1]])
#             legend.append(method)
#     else:
#         for method in front_positions:
    
#             plot_data.append([timeset[1:],(np.asarray(front_positions[method][1:])-np.asarray(front_positions[method][:-1]))/timestep])
#             legend.append(method)
            
#     mplt.graph1d(plot_data,
#                  color=2*mplt.mypalette[:len(front_positions)+1],
#                  legend=legend,
#                  linestyle=linestyle,
#                  axlabels=["",r"$\mathbf{\nu}_{\sigma}(t)$"],
#                  xticks=[[timeset[0],timeset[-1]],[r"$t_0$",r"$t_{\mathrm{max}}$"]],
#                  #yticks=[[lambda_/np.sqrt(timeset[0]),lambda_/np.sqrt(timeset[-1])],[r"$\mathbf{\nu}_{\sigma}(t_0)$",r"$\mathbf{\nu}_{\sigma}(t_{\mathrm{max}})$"]],
#                  yticks=[[],[]],
#                  savefig={"width":345./2,
#                           "name":'./out/fig/'+str(DIM)+'d/front_vel.pdf'
#                  },
#     )

#==================================
