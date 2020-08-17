import matplotlib as mpl
# necessary for plotting in Snehurka cluster:
#mpl.use('Agg')
import matplotlib.pyplot as plt

# Use 'ggplot' style
plt.style.use('seaborn-whitegrid')
# Use LaTeX fonts
#plt.style.use('tex')
#-----------------
# or equivalently (when 'tex' style sheet N/A):
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}

plt.rcParams.update(tex_fonts)
#-----------------

# Custom colors:
mypalette=[[125/256,0,0],[0,0,125/256],[100/256,151/256,191/256],[159/256,203/256,238/256]]
myred=[125/256,0,0]
myblue=[0,0,125/256]

def set_size(width, ratio=(5**.5-1)/2, fraction=1, subplots=(1, 1)):
    """ Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    elif width == 'IJHMT':
        width_pt = 345.
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def graph1d(data,**figprm):
    """ Create a figure and a set of subplots representing one-dimensional plot.

    Parameters
    ----------
    data: array of arrays
            Array of one-dimensional arrays containing data
    Returns
    -------
    fig: Figure matplotlib object
            Dimensions of figure in inches
    ax: Axes matplotlib object
    """
    fig, ax = plt.subplots(1,1)

    # Plot the figure
    for i,xy in enumerate(data):
        try:
            color=figprm.get("color",None)[i]
        except  TypeError:
            color=None
        try:
            marker=figprm.get("marker",None)[i]
        except  TypeError:
            marker=None
        try:
            linestyle=figprm.get("marker",None)[i]
        except  TypeError:
            linestyle=None
        ax.plot(xy[0],xy[1],color=color,marker=marker,linestyle=linestyle)
    # Axes labels
    if "axlabels" in figprm:
        ax.set_xlabel(figprm["axlabels"][0])
        ax.set_ylabel(figprm["axlabels"][1])

    # Set xticks:
    if "xticks" in figprm:
        ax.set_xticks(figprm["xticks"][0])
        ax.set_xticklabels(figprm["xticks"][1])

    # Set yticks:
    if "yticks" in figprm:
        ax.set_yticks(figprm["yticks"][0])
        if len(figprm["yticks"])>1:
            ax.set_yticklabels(figprm["yticks"][1])
        #plt.yticks(figprm["yticks"][0],figprm["yticks"][1],rotation='vertical', va='center')
        plt.yticks(figprm["yticks"][0],rotation='vertical', va='center')

    if "ylim" in figprm:
        ax.set_ylim(bottom=figprm["ylim"].get("bottom",None),
                    top=figprm["ylim"].get("top",None)
        )

    # Set title
    ax.set_title(figprm.get("title",""))

    # Set legend
    ax.legend(figprm.get("legend",None), frameon=True, fancybox=False, borderaxespad=0.)
    
    # Save figure
    if "savefig" in figprm:
        ratio=figprm["savefig"].get("ratio",(5**.5-1)/2)
        fig.set_size_inches(set_size(figprm["savefig"]["width"],ratio=ratio),forward=True)
        fig.savefig(figprm["savefig"]["name"], format='pdf', bbox_inches='tight', transparent=False)
    return fig, ax
