import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

# Behavior color scheme
LOCOMOTION_COLOR = "Blue"
POSITION_COLOR = "Orange"
LICK_COLOR = "Red"
PUPIL_COLOR = "Purple"
WHISKER_COLOR = "Green"
FLUORESCENCE_COLOR = "Black"
INDIVIDUAL_FLUORESCENCE_COLOR = "gray"
SUBTRACT_COLOR = "dimgray"

# Timeline color scheme
PUFF_COLOR = "lightgray"
BLANK_COLOR = "lightgray"
WATER_COLOR = "cornflowerblue"
NOWATER_COLOR = "orangered"
BUZZER_COLOR = "darkorange"
DARK_PUFF_COLOR = "gray"
DARK_BLANK_COLOR = "gray"
DARK_WATER_COLOR = "blue"
DARK_NOWATER_COLOR = "red"
DARK_BUZZER_COLOR = "chocolate"

# Event color scheme
REWARD_TRIAL_COLOR = "#298c8c"
OMISSION_TRIAL_COLOR = "#f1a226"

# Grand color scheme
GRAND_COLOR_SCHEME = {
    "VerticalPuffOn": PUFF_COLOR,
    "Puff": PUFF_COLOR,
    "HorizontalPuffOn": PUFF_COLOR,
    "BlankOn": BLANK_COLOR,
    "Blank": BLANK_COLOR,
    "WaterOn": WATER_COLOR,
    "NoWaterOn": NOWATER_COLOR,
    "BuzzerOn": BUZZER_COLOR,
}

TABLEAU_10 = (
    "#1A4E86", '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
    '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC'
)

def plot_colortable(colors, *, ncols=4, sort_colors=True):
    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        names = sorted(
            colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
    else:
        names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-margin)/height)
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )

    return fig

def show_colortable():
    plot_colortable(mcolors.CSS4_COLORS)
    plt.show()


if __name__ == "__main__":
    show_colortable()