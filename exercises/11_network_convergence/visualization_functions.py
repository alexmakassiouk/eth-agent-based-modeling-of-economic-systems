import matplotlib.pyplot as plt
import networkx as nx


def plot_activity(activities, ax=None, height=0, alpha=0.05, size=100):
    """Plot an activity ribbon (e.g. when a link was created)"""
    if ax is None:
        ax = __provide_missing_ax
    ax.scatter(activities, [height] * len(activities), marker="|", alpha=alpha, s=size)


def plot_evolution(
    world,
    ax=None,
    features=("welfare", "density", "clustering"),
    show_activities=True,
):
    """Show in graph a summary of the evolution"""
    # to determine where to put the ribbon
    if ax is None:
        ax = __provide_missing_ax()
    largest_value = 0
    poling_times = sorted(world.measurements.keys())
    for feature in features:
        values = [world.measurements[t][feature] for t in poling_times]
        largest_value = max(largest_value, max(values))
        ax.plot(poling_times, values, label=feature)

    if show_activities:
        plot_activity(world.deletions.keys(), ax, height=-.02 * largest_value, alpha=.4)
        plot_activity(
            world.insertions.keys(), ax, height=1.02 * largest_value, alpha=.4
        )
    plt.legend()


def plot_network(world, ax=None):
    """Plot the networkx graph for the given World object"""
    if ax is None:
        ax = __provide_missing_ax()
    net = world.net
    pos = nx.drawing.layout.fruchterman_reingold_layout(
        net
    )
    nodelist = [a.uid for a in world.schedule.agents]
    colors = [a.utility(a.subgraph()) for a in world.schedule.agents]
    nx.draw_networkx(net, nodelist=nodelist, node_color=colors, pos=pos, ax=ax)

    # norm = plt.matplotlib.colors.Normalize(vmin=min(colors), vmax=max(colors))
    # scalar_map = plt.cm.ScalarMappable(norm=norm)
    # scalar_map._A = []
    # ax.figure.colorbar(scalar_map)
    ax.axis("off")


def __provide_missing_ax():
    return plt.subplot(1, 1, 1)
