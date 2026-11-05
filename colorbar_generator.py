import matplotlib.pyplot as plt
import matplotlib as mpl

from kitchen.settings.fluorescence import DF_F0_SIGN

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 5

def plot_colorbar(cmap, vmin, vmax, label, figsize, orientation="vertical", ticks=None):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # norm = mpl.colors.SymLogNorm(linthresh=10., vmin=vmin, vmax=vmax)
    cbar = mpl.colorbar.ColorbarBase(
        ax, cmap=plt.get_cmap(cmap), norm=norm, orientation=orientation,
        )
    cbar.ax.tick_params(
        length=1,   # shorter tick marks
        pad=1       # smaller gap between ticks and tick labels
    )
    cbar.set_label(label, labelpad=1)
    cbar.outline.set_visible(False)
    # ax.set_xlabel(label)
    if ticks is not None:
        cbar.set_ticks(ticks)
    elif orientation == "vertical":
        cbar.set_ticks([])
        cbar.ax.text(0.5, 1.02, str(vmax),
             ha='center', va='bottom',
             transform=cbar.ax.transAxes)

        cbar.ax.text(0.5, -0.02, str(vmin),
                    ha='center', va='top',
                    transform=cbar.ax.transAxes)

    fig.savefig("colorbar.png", dpi=500, transparent=True)

if __name__ == "__main__":

    # plot_colorbar(cmap="RdYlBu_r", vmin=-0.5, vmax=0.5, label="", figsize=(0.6, 0.25),
    #               ticks=[-0.5, 0, 0.5], orientation="horizontal")
    plot_colorbar(cmap="coolwarm", vmin=-1, vmax=1, label="", figsize=(0.14, 0.4),
                  ticks=None, orientation="vertical")
