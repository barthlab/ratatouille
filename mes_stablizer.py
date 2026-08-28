from pathlib import Path
import os
import caiman as cm
from caiman.motion_correction import MotionCorrect

raw_tif = Path(
    r"C:\Users\maxyc\PycharmProjects\Ratatouille\ingredients\PassivePuff_HighFreqImaging\HighFreqImaging_202607\M022_TFR_10M\20260701_ROI1\TIFF_20260701_TFR_10M_HFI_day3_ROIs1_Recording002_cropped.tif"
)

out_dir = raw_tif.parent
out_dir.mkdir(exist_ok=True)

out_tif = out_dir / (raw_tif.stem + "_caiman_corrected.tif")

if not raw_tif.exists():
    raise FileNotFoundError(raw_tif)

if raw_tif.resolve() == out_tif.resolve():
    raise RuntimeError("Refusing to overwrite raw TIFF.")

print("Current working directory:", os.getcwd())
print("Raw TIFF:", raw_tif)
print("Output TIFF will be:", out_tif)

mc = MotionCorrect(
    [str(raw_tif)],
    max_shifts=(40, 40),
    pw_rigid=True,
    strides=(16, 64),
    overlaps=(8, 16),
    max_deviation_rigid=5,
    border_nan="copy"
)

mc.motion_correct(save_movie=True)

corrected_mmap = mc.fname_tot_els if mc.pw_rigid else mc.fname_tot_rig

print("CaImAn corrected mmap:", corrected_mmap)

mov_corr = cm.load(corrected_mmap)
mov_corr.save(str(out_tif))

print("Saved corrected TIFF:", out_tif)