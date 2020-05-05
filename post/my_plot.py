import matplotlib as mpl
import matplotlib.pyplot as plt

# Use 'ggplot' style
plt.style.use('ggplot')
# Use LaTeX fonts
plt.style.use('tex')

def set_size(width, fraction=1, subplots=(1, 1)):
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

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

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
    for xy in data:
        ax.plot(xy[0],xy[1])
    # Axes labels
    if "axlabels" in figprm:
        ax.set_xlabel(figprm["axlabels"][0])
        ax.set_ylabel(figprm["axlabels"][1])
    # Title
    if "title" in figprm:
        ax.set_title(figprm["title"])
    # Legend
    if "legend" in figprm:
        ax.legend(figprm["legend"])
    # Save figure
    if "savefig" in figprm:
        fig.set_size_inches(set_size(figprm["savefig"]["width"]))
        fig.savefig(figprm["savefig"]["name"], format='pdf', bbox_inches='tight', transparent=False)
    return fig, ax
