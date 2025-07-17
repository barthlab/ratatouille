from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np

from kitchen.plotter.style_dicts import REFERENCE_LINE_STYLE


TICK_PAIR = namedtuple("TICK_PAIR", ["location", "label", "color"], defaults=["", "black"])

def add_new_yticks(ax: plt.Axes, new_ticks: list[TICK_PAIR] | TICK_PAIR, add_ref_lines: bool = True):    
    if isinstance(new_ticks, TICK_PAIR):
        new_ticks = [new_ticks]
    num_new_ticks = len(new_ticks)

    # Get current ticks and labels
    current_ticks = ax.get_yticks()
    current_labels = [label.get_text() for label in ax.get_yticklabels()]

    # Add new ticks and labels to the current ticks and labels
    for tick in new_ticks:
        if tick.location in current_ticks:
            continue
        current_ticks = np.append(current_ticks, tick.location)
        current_labels.append(tick.label)

    # Set the new ticks and labels on the axis
    ax.set_yticks(ticks=current_ticks, labels=current_labels)
    for tick_instance, new_tick in zip(ax.get_yticklabels()[-num_new_ticks:], new_ticks):
        tick_instance.set_color(new_tick.color)
        ax.axhline(new_tick.location, **REFERENCE_LINE_STYLE)
