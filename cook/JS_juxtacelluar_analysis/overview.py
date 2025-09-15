import logging
from os import path

from kitchen.loader.general_loader_interface import load_dataset



logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n')
handler.setFormatter(formatter)
logger.addHandler(handler)


def main():
    for dataset_name in ("PV_JUX", "PYR_JUX", "SST_JUX", "SST_WC"):

        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                               recipe="default_ephys")
        dataset.status(save_path=path.join(path.dirname(__file__), f"status_report_{dataset_name}.xlsx"), row_level="cellsession")


if __name__ == "__main__":
    main()