from matplotlib import pyplot as plt


def access_twinx(ax: plt.Axes) -> plt.Axes:
    all_siblings = ax.get_shared_x_axes().get_siblings(ax)
    if len(all_siblings) == 2:
        ax_twin = all_siblings[0]
    elif len(all_siblings) == 1:
        ax_twin = ax.twinx()
    else:
        raise ValueError(f"Multiple twin axes found for ax {ax}: {all_siblings}")
    return ax_twin


def access_twiny(ax: plt.Axes) -> plt.Axes:
    all_siblings = ax.get_shared_y_axes().get_siblings(ax)
    if len(all_siblings) == 2:
        ax_twin = all_siblings[0]
    elif len(all_siblings) == 1:
        ax_twin = ax.twiny()
    else:
        raise ValueError(f"Multiple twin axes found for ax {ax}: {all_siblings}")
    return ax_twin