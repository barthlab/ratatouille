import warnings

from kitchen.loader.ephys_loader import cohort_loader
from kitchen.video import custom_extraction, format_converter

warnings.filterwarnings("ignore")

def pre_conversion():
    hft_data_path = r"C:\Users\maxyc\PycharmProjects\Ratatouille\ingredients\PassivePuff_JuxtaCelluar_FromJS\SST_EXAMPLE\video"
    format_converter.stack_tiff_to_video(hft_data_path)
   

if __name__ == "__main__":
    pre_conversion()
    data_set = cohort_loader("PassivePuff_JuxtaCelluar_FromJS", "SST_EXAMPLE")
    print(data_set)
    # custom_extraction.default_collection(data_set, format=".mp4")
