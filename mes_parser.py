from pathlib import Path
import os
from glob import glob
from typing import List

import tifffile as tiff
import numpy as np
from scipy.io import loadmat, whosmat
import pandas as pd


def search_pattern_file(pattern: str, search_dir: str) -> List[str]:
    recursive_path = os.path.join(search_dir, '**', pattern)
    return list(glob(recursive_path, recursive=True))


def write_normal_dataframe(df: pd.DataFrame, sheet_name: str, save_path: str):
    with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        worksheet = writer.sheets[sheet_name]
        worksheet.set_column('A:Z', 20)
        worksheet.set_row(0, 20)
    print(f"Dataframe saved to {save_path}")



def save_multipage_tiff(arr: np.ndarray, path: str, compression=None):
    if arr.dtype != np.uint16:
        raise TypeError(f"Expected uint16, got {arr.dtype}")
    if arr.ndim != 3:
        raise ValueError("Expected a 3D array (Y, X, Z), got {arr.shape}.")
    arr = np.moveaxis(arr, -1, 0)
    tiff.imwrite(
        path,
        arr,                    # (Z, Y, X) becomes pages
        photometric="minisblack",
        compression=compression # optional; use None to disable
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Saved {arr.shape} array to {path}")



def parse_mes(p: Path):
    print(f"Parsing {p}...")
    dir_name, file_name = os.path.split(p)
    variable_list = whosmat(p)
    shapes = {var_name: var_shape for var_name, var_shape, var_type in variable_list}
    for var_name, var_shape, var_type in variable_list:
        print(var_name, var_shape, var_type)
        
        # Dfxxxxx: information, struct, (x, 1) shape
        # Ifxxxxx_0001: image data, (width, total_line) shape, uint16
        if var_name.startswith("D"):  
            # corresponding image data
            image_var_name = var_name.replace("D", "I") + "_0001"
            if shapes[image_var_name][1] < shapes[image_var_name][0] * 10:  # not likely a recording
                continue
            
            # recording index
            recording_idx = int(var_name[2:])
            print(f"Found recording {recording_idx} with shape {shapes[image_var_name]}, start parsing...")
            recording_file_name = file_name.replace(".mes", f"_Recording{recording_idx:03d}")
            Df = loadmat(p, variable_names=[var_name], squeeze_me=True, 
                            struct_as_record=False)[var_name][0]
            # events time
            dt = Df.DIs.x[1]
            ttl = Df.DIs.y

            # image shapes
            width_num, total_line_pixel_num = Df.DIMS
            transverse_pixel_num = Df.TransversePixNum
            start_pixel_index = Df.Clipping.savedHeightBegin
            end_pixel_index = Df.Clipping.savedHeightEnd
            assert Df.Clipping.savedWidthBegin == 1 and Df.Clipping.savedWidthEnd == width_num, \
                f"Only support full width clipping, but got {Df.Clipping.savedWidthBegin} and {Df.Clipping.savedWidthEnd}, Expected 1 and {width_num}"
            assert total_line_pixel_num == (end_pixel_index - start_pixel_index + 1), \
                f"Total line pixel number mismatch, got {total_line_pixel_num} from DIMS, but got {end_pixel_index - start_pixel_index + 1} from Clipping"

            # rounded image shape
            rounded_start_pixel_index = int(np.ceil((start_pixel_index - 1) / transverse_pixel_num) * transverse_pixel_num) + 1
            rounded_end_pixel_index = int(np.floor(end_pixel_index / transverse_pixel_num) * transverse_pixel_num)
            assert ((rounded_start_pixel_index - 1) % transverse_pixel_num == 0) and (rounded_end_pixel_index % transverse_pixel_num == 0), \
                f"Miscalculated shape: Width {width_num}, Total pixel line {total_line_pixel_num}, Transverse pixel {transverse_pixel_num}, "\
                f"Start pixel {start_pixel_index}, End pixel {end_pixel_index}, Rounded start pixel {rounded_start_pixel_index}, Rounded end pixel {rounded_end_pixel_index}"
            n_frame = int((rounded_end_pixel_index - rounded_start_pixel_index + 1) / transverse_pixel_num)

            image_data = loadmat(p, variable_names=[image_var_name], squeeze_me=True, 
                            struct_as_record=False)[image_var_name]
            # reshape image data
            assert image_data.shape == (width_num, total_line_pixel_num), \
                f"Expected image shape {(width_num, total_line_pixel_num)}, but got {image_data.shape}"
            start_offset = rounded_start_pixel_index - start_pixel_index
            end_offset = rounded_end_pixel_index - end_pixel_index if rounded_end_pixel_index != end_pixel_index else None
            reshaped_image_data = image_data[:, start_offset:end_offset].reshape(width_num, transverse_pixel_num, n_frame, order='F').transpose(1, 0, 2)
            
            # save image data
            save_path = os.path.join(dir_name, "TIFF_" + recording_file_name + ".tif")
            save_multipage_tiff(reshaped_image_data, save_path)

            # save recording info
            half_frame_pixel_num = int(transverse_pixel_num // 2)
            start_frame_tick = (rounded_start_pixel_index + half_frame_pixel_num) * width_num
            end_frame_tick = (rounded_end_pixel_index - half_frame_pixel_num) * width_num
            recording_info = {
                "Event t (ms)": (ttl[0] - 1) * dt,
                "Event Tag": ttl[1],
                "Event Tick": ttl[0],
                "Tick dt (ms)": dt,
                "First Frame t (ms)": (start_frame_tick - 1) * dt,
                "Last Frame t (ms)": (end_frame_tick - 1) * dt,
                "Frame Rate (Hz)":  (n_frame - 1) / ((end_frame_tick - start_frame_tick) * dt / 1000),
                "Width (pixel)": width_num,
                "Height (pixel)": transverse_pixel_num,
                "Frame #": n_frame,
                " ": None,
                "Saved Total Line #": total_line_pixel_num,
                "Saved Height Begin": start_pixel_index,
                "Saved Height End": end_pixel_index,
                "Saved Complete Frame #": int(total_line_pixel_num // transverse_pixel_num),
                "Rounded Height Begin": rounded_start_pixel_index,
                "Rounded Height End": rounded_end_pixel_index,
                "Rounded Complete Frame #": n_frame,
                "Half Frame Pixel #": half_frame_pixel_num,
                "Rounded Start Frame Tick": start_frame_tick,
                "Rounded End Frame Tick": end_frame_tick,
            }
            for k, v in recording_info.items():
                recording_info[k] = pd.Series(v)
            save_path = os.path.join(dir_name, "INFO_" + recording_file_name + ".xlsx")
            write_normal_dataframe(pd.DataFrame(recording_info), f"Recording{recording_idx:03d}", save_path)



def parse_all_mes_under_dir(dir_path: str):
    all_mes_files = search_pattern_file("*.mes", dir_path)
    print(f"Found {len(all_mes_files)} .mes files under {dir_path}, start parsing...")
    for p in Path(dir_path).rglob("*.mes"):
        parse_mes(p)
    print("Parsing complete!")
    

if __name__ == "__main__":
    parse_all_mes_under_dir(r"C:\Users\maxyc\PycharmProjects\Ratatouille\ingredients\PassivePuff_HighFreqImaging\HighFreqImaging_202602")