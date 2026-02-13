import os.path as path
import logging

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.plotter.decorators.default_decorators import default_exit_save, default_plt_param
from kitchen.plotter.plotting_manual import PlotManual


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# handler = logging.FileHandler(path.join(path.dirname(__file__), "overview.log"), mode="w")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n')
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING) 


def main():
    dataset = load_dataset(template_id="HeadFixedTraining", cohort_id="SensoryPrediction_202510-11", 
                           recipe="default_behavior_only", name="sensory_prediction")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx")) 
    plot_manual = PlotManual(locomotion=True, whisker=True, pupil=True, saccade=True)  

    # dataset = dataset.subset(session_id="20251009_RSS4_cuedpuff_4_cuedpuff90")
    # dataset = dataset.subset(session_id="20251127_SQC_1M_CP_omit_day9_3_CP_omit")
    # dataset = dataset.subset(session_id="20251217_SRW_1M_CP_omit2_day8_5_CP_omit")
    # dataset = dataset.subset(session_id="20251217_SUH_5M_CL_omit2_day8_3_CL_omit")
    # dataset = dataset.subset(session_id="20251025_SJC7F_cuedpuff_2_cuedpuff90")
    # dataset = dataset.subset(session_id="20251127_SQG_5F_CuedPuff90_day8_5_CuedPuff90")
    dataset = dataset.subset(session_id="20251125_SQC_1M_CuedPuff90_day7_5_CuedPuff90")
    print(dataset.select("session"))

    import matplotlib.pyplot as plt
    from kitchen.plotter.ax_plotter import basic_plot
    default_plt_param()

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.), constrained_layout=True)
    ax.tick_params(axis='y', labelleft=True) 
    ax.set_yticks([])
    plotter = basic_plot.flat_view(ax=ax, datasets=dataset.select("session"), plot_manual=plot_manual)
    progress = next(plotter)
    offset = (0.5, 0.5, 1, 2, 0.5, 0, 0)
    while True:
        try:
            progress += plotter.send(progress + offset[0])
        except StopIteration:
            break
        offset = offset[1:]
        
    # ax.set_xlim(50, 125)
    # ax.set_xlim(280, 350)
    # ax.set_xlim(265, 390)
    # ax.set_xlim(500, 575)
    # ax.set_xlim(385, 445)
    ax.set_xlim(215, 285)
    ax.spines[['top', 'right']].set_visible(False)
    save_path = path.join(path.dirname(__file__), "single_session_example7.png")
    default_exit_save(fig, save_path)

if __name__ == "__main__":
    main()