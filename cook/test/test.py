import sys
import os
import warnings

import kitchen.loader.hierarchical_loader as hier_loader

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print(hier_loader.cohort_loader(template_id="one_day", cohort_id="SingleCellCalciumImaging"))
    print(hier_loader.cohort_loader(template_id="SAT", cohort_id="VIPcreAi148_FromMo"))
    print(hier_loader.cohort_loader(template_id="random", cohort_id="HeadFixedTraining"))