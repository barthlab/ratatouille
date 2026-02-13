from collections import namedtuple
from typing import Dict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from kitchen.plotter import style_dicts
from kitchen.plotter.style_dicts import LEGEND_STYLE, REFERENCE_LINE_STYLE


TICK_PAIR = namedtuple("TICK_PAIR", ["location", "label", "color"], defaults=["", "black"])

def add_new_yticks(ax: plt.Axes, new_ticks: list[TICK_PAIR] | TICK_PAIR, add_ref_lines: bool = False):    
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
        if add_ref_lines:
            ax.axhline(new_tick.location, **REFERENCE_LINE_STYLE)


def add_line_legend(ax: plt.Axes, lines: Dict[str, dict], **kwargs):
    legend_handles = [Line2D([0], [0], label=label, **style) for label, style in lines.items()]
    ax.legend(handles=legend_handles, **(LEGEND_STYLE | kwargs))


def add_textonly_legend(ax: plt.Axes, texts: Dict[str, dict], **kwargs):
    labels = list(texts.keys())
    handles = [Line2D([0], [0], visible=False) for _ in labels]

    leg = ax.legend(handles=handles, labels=labels, 
        handlelength=0, handletextpad=0, **(LEGEND_STYLE | kwargs))

    for text_obj, label_text in zip(leg.get_texts(), labels):
        properties = texts[label_text]
        text_obj.update(properties)


def emphasize_yticks(ax: plt.Axes,):
    ax.tick_params(axis='y', which='major', length=10, width=1.5)
    ax.tick_params(axis='y', which='minor', length=5, width=1,)
    

def add_labeless_yticks(ax: plt.Axes, MajorTicks: list[float] | np.ndarray, MinorTicks: list[float] | np.ndarray=[], _hlines: bool = False):
    current_ticks = ax.get_yticks()    
    current_labels = [label.get_text() for label in ax.get_yticklabels()]
    
    tick_dict = dict(zip(current_ticks, current_labels))
    for tick in MajorTicks:
        if tick not in tick_dict:
            tick_dict[tick] = ""
    all_ticks = sorted(tick_dict.keys())
    all_labels = [tick_dict[t] for t in all_ticks]

    ax.set_yticks(all_ticks)
    ax.set_yticklabels(all_labels)

    ax.set_yticks(MinorTicks, minor=True)

    if _hlines:
        for tick in MajorTicks:
            ax.axhline(tick, **style_dicts.TICK_LINE_STYLE)
    