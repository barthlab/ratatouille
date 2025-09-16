import logging
from os import path

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.plotter.macros.basic_macros import session_overview
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.structure.hierarchical_data_structure import Session



logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# handler = logging.FileHandler(path.join(path.dirname(__file__), "overview.log"), mode="w")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n')
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def main():
    for dataset_name in ("PV_JUX", "PYR_JUX", "SST_JUX", "SST_WC"):

        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                               recipe="default_ephys")
        # dataset.status(save_path=path.join(path.dirname(__file__), f"status_report_{dataset_name}.xlsx"), row_level="cellsession")

        plot_manual_raw = PlotManual(potential=True)
        plot_manual_spike4Hz = PlotManual(potential=4.)
        plot_manual_spike300Hz = PlotManual(potential=300.)
        for session_node in dataset.select(hash_key="cellsession"):
            print(session_node)
            session_overview(session_node, plot_manual=plot_manual_raw, special_keyword="raw")
            session_overview(session_node, plot_manual=plot_manual_spike4Hz, special_keyword="spike4Hz")
            session_overview(session_node, plot_manual=plot_manual_spike300Hz, special_keyword="spike300Hz")


if __name__ == "__main__":
    main()