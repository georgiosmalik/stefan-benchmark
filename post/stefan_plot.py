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
from math import floor, log10

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

    theta_max = 0
    
    for i,t in enumerate(timeset):
        x_range=data["temp_dist"][str(t)]["analytic"][0]
        y_range=data["temp_dist"][str(t)]["analytic"][1]

        theta_max = max(theta_max,y_range[0])

        plt.plot(x_range,
                 y_range,
                 lw=2.,
                 color=mplt.mypalette[i+1],
                 marker=linestyle["analytic"]["marker"]
        )

        if t in timeset[1:-1]:
            # Create legend with time steps:
            stamp=(t-timeset[0])/(timeset[-1]-timeset[0])
            names.append(r"$t_"+str(i)+"{=}t_0{+}"+str(f"{stamp:.1f}")+"\!\cdot\! \\tau$")
            xticks[0].append(2*lambda_*np.sqrt(t))
            xticks[1].append(r"$s(t_"+str(i)+")$")

        methodplots=()
            
        for method in methods:

            x_range=data["temp_dist"][str(t)][method][0]
            y_range=data["temp_dist"][str(t)][method][1]

            plot,=plt.plot(x_range,
                           y_range,
                           label=method,
                           linestyle='None',
                           color=mplt.mypalette[i+1],
                           marker=linestyle[method]["marker"],
                           markersize=3,
                           markevery=(0.+len(methodplots)*0.1/len(methods),0.1)
            )
            methodplots=methodplots+(plot,)
            
        plots.append(methodplots)

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

    # Color gridlines for melt. front position:
    for i,xgline in enumerate(ax.get_xgridlines()[1:len(timeset)+1]):
        xgline.set_color(mplt.mypalette[i+1])
        xgline.set_linestyle("dotted")
        
    ax.set_yticks([prm.theta_m, theta_max])
    ax.set_yticklabels([r"$\theta_\mathrm{m}$",r"$"+str(int(theta_max))+"\,\mathrm{K}$"])

    ax.get_yticklabels()[-1].set_rotation('vertical')

    # Color gridlines for melt. front position:
    ax.get_ygridlines()[0].set_linestyle("dotted")

    #------------------------------
    # Save the figure:
    h_max=data["disc_params"]["h_max"]
    
    eps=data["disc_params"]["eps"]
    h_eps=data["disc_params"]["h_eps"]
    c_eps=data["disc_params"]["C_eps"]
    
    dt=data["disc_params"]["dt"]
    c_cfl=data["disc_params"]["C_CFL"]

    fig.set_size_inches(mplt.set_size(2*345./3),forward=True)
    fig.savefig('./out/fig/'+str(DIM)+'d/temp_dist_(eps='+'{0:>1.1e}'.format(eps)+'(h_eps='+'{0:>1.2e}'.format(h_eps)+',C_eps='+'{0:>1.1e}'.format(c_eps)+'),h_max='+'{0:>1.2e}'.format(h_max)+',dt='+'{0:>1.2e}'.format(dt)+'(C_CFL='+'{0:>1.1e}'.format(c_cfl)+').pdf',
                format='pdf',
                bbox_inches='tight',
                transparent=False
    )
    #--------------------------------------------

# Graph melting front position:
def graph_front_pos():

    data=DATA_PY

    lambda_=data["problem_params"]["lambda"]
    timeset=data["problem_params"]["sim_timeset"]
    r1 = data["problem_params"]["r1"]
    r2 = data["problem_params"]["r2"]
    methods=list(data["front_pos"].keys())

    plot_data=[[timeset,2*lambda_*np.sqrt(timeset)]]
    legend=['analytic']

    tau=timeset[-1]-timeset[0]

    # We compute with a fixed timestep
    timestep=timeset[1]-timeset[0]
    
    for method in methods:
        front_positions=data["front_pos"][method]
        plot_data.append([timeset,np.asarray(front_positions)])
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
                 yticks=[[r1,r2],[r"$r_1$",r"$r_2$"]],
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

    vel_max = lambda_/np.sqrt(timeset)[0]

    try:
        vel_max_exp = int(np.floor(np.log10(vel_max)))
    except OverflowError:
        vel_max_exp = 0

    vel_max_man = vel_max/10**vel_max_exp

    for method in methods:
        
        front_positions=np.asarray(data["front_pos"][method])

        num_spline_points = 50

        if num_spline_points >= len(timeset):
            spline_timeset=timeset
        else:
            idx_spline=np.round(np.linspace(num_spline_points,len(timeset)-1,num_spline_points)).astype(int)
            spline_timeset = np.append(timeset[0:num_spline_points],timeset[idx_spline])
            front_pos_spline = np.append(front_positions[0:num_spline_points],front_positions[idx_spline])
        
        pos_spline = CubicSpline(spline_timeset, front_pos_spline)
        
        vel_spline = pos_spline(timeset, 1)
        vel_spline = pos_spline.derivative()
        
        plot_data.append([timeset,vel_spline(timeset)])
        legend.append(method)
            
    # Save the figure:
    mplt.graph1d(plot_data,
                 color=2*mplt.mypalette[:len(methods)+1],
                 legend=legend,
                 linestyle=linestyle,
                 axlabels=["",r"$\mathbf{\nu}_{\sigma}(t)\,[10^{"+str(vel_max_exp)+"}\mathrm{m}\cdot \mathrm{s}^{-1}]$"],
                 xticks=[[timeset[0],timeset[-1]],[r"$t_0$",r"$t_{\mathrm{max}}$"]],
                 yticks=[[0,vel_max],[r"$0$",r"$"+'{0:1.2f}'.format(vel_max_man)+"$"]],
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

        ax2.set_xlabel(r"$C_\epsilon\,[-]$")
        ax2.set_xlim(min(X_C_eps), max(X_C_eps))

        ax2.set_axisbelow(True)

        # Color gridlines for alternative x-axis:
        for xgline in ax2.get_xgridlines():
            xgline.set_linestyle("dotted")
        
        ax.plot(X, fp_err, color = mplt.mypalette[0], label = r'$|s-\overline{s}|/|s|$')
        ax.plot(X, l2_err, color = mplt.mypalette[2], linestyle = 'dotted', lw = 2, label = r'$\|\theta-\overline{\theta}\|_{L^2}/\|\theta\|_{L^2}$')
        #ax.plot(X,linf_err, color = mplt.mypalette[2], linestyle = 'dotted', lw = 2., label = r'$\|\theta-\overline{\theta}\|_{L^\infty}/\|\theta\|_{L^\infty}$')
        ax.set_xlabel(r"$\epsilon\,[K]$")
        ax.invert_xaxis()

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

    for err_type in ['fp_err','l2_err','linf_err']:
        for method in data:
            Z=[]
            Z_log=[]
    
            for deltat in timestep:
                line=[]
    
                for h in meshres:
                    line.append(data[method][h][deltat][err_type])
    
                Z.append(line)
    
            Z=np.asarray(Z)
            Z_log=np.log10(Z)

            # 3d plot:
            fig = plt.figure()

            # Add right padding for zlabel to show properly:
            fig.subplots_adjust(right = 0.8)
            
            ax = fig.gca(projection='3d')
            surf = ax.plot_surface(X_gr,Y_gr,Z_log, linewidth=0.5, alpha=0.9, cmap=mycmp[method])
            
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
            ax.xaxis.get_ticklines()[-1].set_color(color)
            
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks_labels)
            
            ax.yaxis.get_ticklabels()[-1].set_color(color)
            ax.yaxis.get_ticklabels()[-1].set_fontsize(5)
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
