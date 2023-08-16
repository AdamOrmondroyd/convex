import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import anesthetic as ac


def clustering_animation(chains):

    ns = ac.read_chains(f"chains/{chains}")
    fig, ax = plt.subplots()
    ax.set(xlim=(-1, 1), ylim=(-1, 1))

    def animate(i):
        print(i)
        ax.clear()
        ax.set_title(f"{chains} {i}/{len(ns)}")
        lp = ns.live_points(i)
        clusters = np.unique(lp.cluster)
        for c in clusters:
            axres = lp[lp.cluster == c].plot.scatter_2d(0, 1, ax=ax,
                                                        color=f"C{c}",
                                                        label=str(c))
        ax.legend(loc="lower left", frameon=False)
        axres.set(xlim=(-1, 1), ylim=(-1, 1))
        return axres

    ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, len(ns), 10), interval=20)
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15,
                                    metadata=dict(artist='Me'),
                                    bitrate=1800)
    ani.save(f'{chains}.gif', writer=writer)


for chains in sys.argv[1:]:
    clustering_animation(chains)
    plt.show()
