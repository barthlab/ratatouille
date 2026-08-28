import matplotlib.pyplot as plt
import matplotlib as mpl

from kitchen.settings.fluorescence import DF_F0_SIGN

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 6

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
    cbar.set_label(label)
    # ax.set_xlabel(label)
    if ticks is not None:
        cbar.set_ticks(ticks)
    fig.savefig("colorbar.png", dpi=500, transparent=True)

if __name__ == "__main__":

    plot_colorbar(cmap="coolwarm", vmin=-1, vmax=1, label="Correlation", figsize=(0.4, 0.5),
                  ticks=[-1, 0, 1], orientation="vertical")
