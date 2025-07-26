

from functools import partial
from typing import Tuple
import warnings
from kitchen.calculator.basic_metric import AVERAGE_VALUE
from kitchen.calculator.calculator_interface import calculate_metric
from kitchen.configs import routing
from kitchen.configs.naming import get_node_name
from kitchen.operator.select_trials import PREDEFINED_RULES
from kitchen.plotter.ax_plotter.advance_plot import subtract_view
from kitchen.plotter.ax_plotter.basic_bar import pairwise_comparison_bar
from kitchen.plotter.ax_plotter.basic_plot import stack_view
from kitchen.plotter.color_scheme import OMISSION_TRIAL_COLOR, REWARD_TRIAL_COLOR
from kitchen.plotter.decorators.default_decorators import default_style
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.plotter.unit_plotter.unit_trace_advance import SUBTRACT_MANUAL
from kitchen.settings.timeline import ALIGNMENT_STYLE
from kitchen.structure.hierarchical_data_structure import DataSet, Day, Mice



def find_omission_day_id(dataset: DataSet) -> dict[str, Tuple[Day, Day, Day]]:
    """Find the day id of the omission day for each mice."""
    omission_day_id_dict = {}
    for mice_node in dataset.select("mice"):
        assert isinstance(mice_node, Mice)
        
        mice_dataset = dataset.subtree(mice_node)
        day_nodes = [day_node for day_node in mice_dataset.select("day")]
        n_day = len(day_nodes)
        for day_idx in range(n_day - 2):
            before_day = day_nodes[day_idx]
            cur_day = day_nodes[day_idx + 1]
            after_day = day_nodes[day_idx + 2] 
            if len(mice_dataset.subtree(before_day).select(**PREDEFINED_RULES["PuffNoWater"])) == 0 and\
                len(mice_dataset.subtree(cur_day).select(**PREDEFINED_RULES["PuffNoWater"])) > 0:
                omission_day_id_dict[mice_node.mice_id] = (before_day, cur_day, after_day)
    print(f"Found omission days (before, omission, after) for {len(omission_day_id_dict)} mice")
    for mice_id, (_, omission_day, _) in omission_day_id_dict.items():
        print(f"{mice_id}: water omission start at {get_node_name(omission_day)}")
    return omission_day_id_dict


def mice_water_omission_overview(
        dataset: DataSet,
        plot_manual: PlotManual,
):
    omission_day_id_dict = find_omission_day_id(dataset)
    plotting_trial_types = ["PuffWater", "PuffNoWater"]
    subtract_manual = SUBTRACT_MANUAL(name1="Reward", name2="Omission")
    for mice_node in dataset.select("mice"):
        assert isinstance(mice_node, Mice)

        _, omission_day_node, after_day_node = omission_day_id_dict[mice_node.mice_id]
        omission_name, after_name = get_node_name(omission_day_node), get_node_name(after_day_node)
        mice_dataset = dataset.subtree(mice_node)
        for alignment_name, alignment_events in ALIGNMENT_STYLE.items():
            try:
                content_dict = {
                    f"{get_node_name(day_node)}\n{trial_type}": (
                        partial(stack_view, plot_manual=plot_manual, sync_events=alignment_events),
                        mice_dataset.subtree(day_node).select(**PREDEFINED_RULES[trial_type])
                    )
                    for day_node in [omission_day_node, after_day_node]
                    for trial_type in plotting_trial_types
                    }
                content_dict = content_dict | {
                    f"{omission_name}\nSubtraction": (
                        partial(subtract_view, subtract_manual=subtract_manual, plot_manual=plot_manual, sync_events=alignment_events),
                        [mice_dataset.subtree(omission_day_node).select(**PREDEFINED_RULES["PuffWater"]),
                         mice_dataset.subtree(omission_day_node).select(**PREDEFINED_RULES["PuffNoWater"])]
                    ),
                    f"{after_name}\nSubtraction": (
                        partial(subtract_view, subtract_manual=subtract_manual, plot_manual=plot_manual, sync_events=alignment_events),
                        [mice_dataset.subtree(after_day_node).select(**PREDEFINED_RULES["PuffWater"]),
                         mice_dataset.subtree(after_day_node).select(**PREDEFINED_RULES["PuffNoWater"])]
                    ),
                    }
                default_style(
                    mosaic_style=[
                        [f"{omission_name}\n{trial_type}" for trial_type in plotting_trial_types] + [f"{omission_name}\nSubtraction"],
                        [f"{after_name}\n{trial_type}" for trial_type in plotting_trial_types] + [f"{after_name}\nSubtraction"],
                    ],
                    content_dict=content_dict,
                    figsize=(4.5, 3),
                    save_path=routing.default_fig_path(mice_dataset, f"WaterOmission_{{}}_{alignment_name}.png"),
                )
            except Exception as e:
                warnings.warn(f"Cannot plot water omission for {get_node_name(mice_node)} with {alignment_name}: {e}")



def mice_water_omission_summary(
        dataset: DataSet,
        plot_manual: PlotManual,
):
    omission_day_id_dict = find_omission_day_id(dataset)
    subtract_manual = SUBTRACT_MANUAL(color1=REWARD_TRIAL_COLOR, color2=OMISSION_TRIAL_COLOR, name1="Reward", name2="Omission")
    for alignment_name, alignment_events in ALIGNMENT_STYLE.items():
        try:
            content_dict = {
                f"{mice_node.mice_id} {day_name}\nSubtraction": (
                    partial(subtract_view, subtract_manual=subtract_manual, plot_manual=plot_manual, sync_events=alignment_events),
                    [dataset.subtree(mice_node).subtree(day_node).select(**PREDEFINED_RULES["PuffWater"]),
                     dataset.subtree(mice_node).subtree(day_node).select(**PREDEFINED_RULES["PuffNoWater"])]
                )
                for mice_node in dataset.select("mice")
                for day_node, day_name in zip(omission_day_id_dict[mice_node.mice_id][1:], ["Omission day0", "Omission day1"])
                }
            content_dict = content_dict | {
                f"{mice_node.mice_id} Omission day0/1\nSubtraction": (
                    partial(subtract_view, subtract_manual=subtract_manual, plot_manual=plot_manual, sync_events=alignment_events),
                    [dataset.subtree(mice_node).select(
                        day_id=lambda day_id: day_id in [omission_day_id_dict[mice_node.mice_id][1].day_id, omission_day_id_dict[mice_node.mice_id][2].day_id],
                        **PREDEFINED_RULES["PuffWater"]),
                     dataset.subtree(mice_node).select(
                         day_id=lambda day_id: day_id in [omission_day_id_dict[mice_node.mice_id][1].day_id, omission_day_id_dict[mice_node.mice_id][2].day_id],
                         **PREDEFINED_RULES["PuffNoWater"])]
                )
                for mice_node in dataset.select("mice")
                }
            default_style(
                mosaic_style=[[f"{mice_node.mice_id} Omission day0\nSubtraction" for mice_node in dataset.select("mice")],
                              [f"{mice_node.mice_id} Omission day1\nSubtraction" for mice_node in dataset.select("mice")],
                              [f"{mice_node.mice_id} Omission day0/1\nSubtraction" for mice_node in dataset.select("mice")],
                              ],
                content_dict=content_dict,
                figsize=(6, 4.5),
                save_path=routing.default_fig_path(dataset, f"AllMiceWaterOmission_{alignment_name}.png"),
            )
        except Exception as e:
            warnings.warn(f"Cannot plot all mice water omission for with {alignment_name}: {e}")


def water_omission_response_compare(
        dataset: DataSet,
        plot_manual: PlotManual,
):
    omission_day_id_dict = find_omission_day_id(dataset)
    alignment_style = ALIGNMENT_STYLE["Aligned2Reward"]
    
    # get all day0 and day1 dataset
    day0_dataset = sum([dataset.subtree(mice_node).subtree(omission_day_id_dict[mice_node.mice_id][1]) for mice_node in dataset.select("mice")], DataSet(name="Day0"))
    day1_dataset = sum([dataset.subtree(mice_node).subtree(omission_day_id_dict[mice_node.mice_id][2]) for mice_node in dataset.select("mice")], DataSet(name="Day1"))
    day01_dataset = day0_dataset + day1_dataset
    after_reward_avg = partial(AVERAGE_VALUE, segment_period=(0.0, 5.0))
    subtract_manual = SUBTRACT_MANUAL(color1=REWARD_TRIAL_COLOR, color2=OMISSION_TRIAL_COLOR, name1="Reward", name2="Omission")

    # make content_dict 
    for data_name, data_flag in plot_manual._asdict().items():
        if not data_flag or data_name == "timeline":
            continue
        sub_plot_manual = PlotManual(**{data_name: True})
        content_dict = {
            f"{data_name}\n{dataset_label}": (
                partial(pairwise_comparison_bar, 
                        metric_funcs=[
                            partial(calculate_metric, dst_dataset=dataset, data_name=data_name, calc_func=after_reward_avg, sync_events=alignment_style),
                            partial(calculate_metric, dst_dataset=dataset, data_name=data_name, calc_func=after_reward_avg, sync_events=alignment_style),
                        ],
                        bar_styles=[
                            {'label': "Reward", 'color': REWARD_TRIAL_COLOR}, 
                            {'label': "Omission", 'color': OMISSION_TRIAL_COLOR}
                        ],
                        comparison_method="ttest_rel",
                        data_name=data_name,
                        ),
                [sub_dataset.select(**PREDEFINED_RULES["PuffWater"]),
                sub_dataset.select(**PREDEFINED_RULES["PuffNoWater"])]
            )
            for sub_dataset, dataset_label in zip([day0_dataset, day1_dataset, day01_dataset], ["Day0", "Day1", "Day0/1"])
        }
        
        default_style(
            mosaic_style=[[f"{data_name}\nDay0", f"{data_name}\nDay1", f"{data_name}\nDay0/1"],],
            content_dict=content_dict,
            figsize=(4.5, 1.),
            save_path=routing.default_fig_path(dataset, f"WOResponse_{data_name}_bar.png"),
            sharex=False,
        )

        content_dict = {
            f"{mice_node.mice_id}\n{dataset_label}": (
                partial(subtract_view, subtract_manual=subtract_manual, plot_manual=sub_plot_manual, sync_events=alignment_style),
                [sub_dataset.select(mice_id=mice_node.mice_id, **PREDEFINED_RULES["PuffWater"]),
                 sub_dataset.select(mice_id=mice_node.mice_id, **PREDEFINED_RULES["PuffNoWater"])]
            )
            for sub_dataset, dataset_label in zip([day0_dataset, day1_dataset, day01_dataset], ["Day0", "Day1", "Day0/1"])            
            for mice_node in dataset.select("mice")
        }

        default_style(
            mosaic_style=[
                [f"{mice_node.mice_id}\nDay0" for mice_node in dataset.select("mice")],
                [f"{mice_node.mice_id}\nDay1" for mice_node in dataset.select("mice")],
                [f"{mice_node.mice_id}\nDay0/1" for mice_node in dataset.select("mice")],
                ],
            content_dict=content_dict,
            figsize=(1.5 * len(dataset.select("mice")), 3),
            save_path=routing.default_fig_path(dataset, f"WOResponse_{data_name}_traces.png"),
            sharex=False,
        )
