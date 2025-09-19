import sys
import os
import warnings


from kitchen.structure.hierarchical_data_structure import Fov
import kitchen.media.format_converter as format_converter
import kitchen.media.custom_extraction as custom_extraction
import kitchen.loader.two_photon_loader as hier_loader

warnings.filterwarnings("ignore")

def main():
    data_set = hier_loader.naive_loader(template_id="Matt's Head-Fixed Protocol", cohort_id="HeadFixedTraining_FromMatt")
    format_converter.dataset_interface_h264_2_avi(data_set)
    custom_extraction.default_collection(data_set)
   

if __name__ == "__main__":
    main()