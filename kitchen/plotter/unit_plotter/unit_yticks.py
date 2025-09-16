from matplotlib import pyplot as plt

from kitchen.plotter.color_scheme import FLUORESCENCE_COLOR, LICK_COLOR, LOCOMOTION_COLOR, POSITION_COLOR, POTENTIAL_COLOR, PUPIL_COLOR, WHISKER_COLOR
from kitchen.plotter.utils.tick_labels import TICK_PAIR, add_new_yticks
from kitchen.settings.fluorescence import DF_F0_SIGN
from kitchen.utils.value_dispatch import value_dispatch


@value_dispatch
def yticks_combo(command: str, ax: plt.Axes, y_offset: float, ratio: float = 1.0, **kwargs):
    ax.set_yticks([y_offset,], **kwargs)


@yticks_combo.register("locomotion")
def _yticks_combo_locomotion(command: str, ax: plt.Axes, y_offset: float, ratio: float = 1.0, **kwargs):
    add_new_yticks(ax, [TICK_PAIR(y_offset, "Locomotion", LOCOMOTION_COLOR), 
                        TICK_PAIR(y_offset + 2 * ratio, "2 cm/s", LOCOMOTION_COLOR)])
    

@yticks_combo.register("position")
def _yticks_combo_position(command: str, ax: plt.Axes, y_offset: float, ratio: float = 1.0, **kwargs):
    add_new_yticks(ax, [TICK_PAIR(y_offset, "0°", POSITION_COLOR), 
                        TICK_PAIR(y_offset + 0.5 * ratio, "180°", POSITION_COLOR),
                        TICK_PAIR(y_offset + 1 * ratio, "360°", POSITION_COLOR)])
    

@yticks_combo.register("lick")
def _yticks_combo_lick(command: str, ax: plt.Axes, y_offset: float, ratio: float = 1.0, **kwargs):
    add_new_yticks(ax, [TICK_PAIR(y_offset, "Lick", LICK_COLOR),
                        TICK_PAIR(y_offset + 5 * ratio, "5 Hz", LICK_COLOR)])
    

@yticks_combo.register("pupil")
def _yticks_combo_pupil(command: str, ax: plt.Axes, y_offset: float, ratio: float = 1.0, **kwargs):
    add_new_yticks(ax, [TICK_PAIR(y_offset, "Min Pupil", PUPIL_COLOR),
                        TICK_PAIR(y_offset + ratio, "Max Pupil", PUPIL_COLOR)])
    

@yticks_combo.register("whisker")
def _yticks_combo_whisker(command: str, ax: plt.Axes, y_offset: float, ratio: float = 1.0, **kwargs):    
    add_new_yticks(ax, [TICK_PAIR(y_offset, "Min Whisker", WHISKER_COLOR),
                        TICK_PAIR(y_offset + ratio, "Max Whisker", WHISKER_COLOR)])
    

@yticks_combo.register("fluorescence")
def _yticks_combo_fluorescence(command: str, ax: plt.Axes, y_offset: float, ratio: float = 1.0, **kwargs):    
    add_new_yticks(ax, TICK_PAIR(y_offset, "Fluorescence", FLUORESCENCE_COLOR)) 
    add_new_yticks(ax, TICK_PAIR(y_offset + ratio, f"1 {DF_F0_SIGN}", FLUORESCENCE_COLOR))


@yticks_combo.register("potential_jux")
def _yticks_combo_potential_jux(command: str, ax: plt.Axes, y_offset: float, ratio: float = 1.0, **kwargs):    
    add_new_yticks(ax, TICK_PAIR(y_offset, "Jux", POTENTIAL_COLOR)) 
    add_new_yticks(ax, TICK_PAIR(y_offset + ratio, "1 mV", POTENTIAL_COLOR))


@yticks_combo.register("potential_wc")
def _yticks_combo_potential_wc(command: str, ax: plt.Axes, y_offset: float, ratio: float = 1.0, **kwargs):    
    add_new_yticks(ax, TICK_PAIR(y_offset, "Whole Cell", POTENTIAL_COLOR)) 
    add_new_yticks(ax, TICK_PAIR(y_offset + 20 * ratio, "20 mV", POTENTIAL_COLOR))
