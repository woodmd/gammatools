
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def plot_ul(x,y,ax=None,color=None):

    if ax is None: ax = plt.gca()

    color_cycle = ax._get_lines.color_cycle

    color = color_cycle.next()

    style = {'linestyle' : 'None', 'linewidth' : 2}
    if not color is None: style['color'] = color
    ax.errorbar(x,y,xerr=0.125,**style)

    for i in range(len(x)):

        c = FancyArrowPatch((x[i],y[i]), 
                            (x[i],y[i]*0.6), 
                            arrowstyle="-|>,head_width=4,head_length=8",
                            lw=2,color=color)
        ax.add_patch(c)



