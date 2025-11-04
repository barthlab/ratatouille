import logging

import kitchen.plotter.color_scheme as color_scheme
from kitchen.plotter.macros.JS_juxta_data_macros_SummaryMetric import PlainHeatmap_PSTH, SummaryMetric_PSTH, SummaryMetric_LFP, SummaryMetric_Waveform
import kitchen.plotter.style_dicts as style_dicts
from kitchen.plotter.macros.JS_juxta_data_macros_Settings import COHORT_COLORS



logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# handler = logging.FileHandler(path.join(path.dirname(__file__), "overview.log"), mode="w")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n')
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger('matplotlib').setLevel(logging.WARNING)



def main():
    SummaryMetric_PSTH()
    SummaryMetric_LFP()
    SummaryMetric_Waveform() 
    PlainHeatmap_PSTH()
if __name__ == "__main__":
    main()