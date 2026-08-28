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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tiff.imwrite(
        path,
        arr,                    # (Z, Y, X) becomes pages
        photometric="minisblack",
        compression=compression # optional; use None to disable
    )
    print(f"Saved {arr.shape} array to {path}")


def save_zstack_tiff(arr: np.ndarray, stack_info: dict, path: str, compression=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    assert arr.shape == (stack_info["DepthPixelNum"], stack_info["HeightPixelNum"], stack_info["WidthPixelNum"],), \
        f"Expected shape {(stack_info['DepthPixelNum'], stack_info['HeightPixelNum'], stack_info['WidthPixelNum'])}, got {arr.shape}"
    
    tiff.imwrite(
        path,
        arr[:, None, :, :],
        metadata={
            "axes": "ZCYX",
            "PhysicalSizeX": stack_info["WidthStep"],
            "PhysicalSizeY": stack_info["HeightStep"],
            "PhysicalSizeZ": stack_info["DepthStep"],
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZUnit": "µm",
        }
    )
    print(f"Saved {arr.shape} array to {path}")


def parse_mes(p: Path):
    print(f"Parsing {p}...")
    dir_name, file_name = os.path.split(p)
    variable_list = whosmat(p)
    shapes = {var_name: var_shape for var_name, var_shape, var_type in variable_list}
    for var_name, var_shape, var_type in variable_list:
        # Dfxxxxx: information, struct, (x, 1) shape
        # Ifxxxxx_0001: image data, (width, total_line) shape, uint16
        if not var_name.startswith("D"):  
            continue
        print(var_name, var_shape, var_type)
        
        # corresponding image data
        image_var_name = var_name.replace("D", "I") 
        corresponding_image_names = sorted([
            name for name in shapes.keys() 
            if name.startswith(image_var_name)
            and len(name) == len(image_var_name) + 5  # "_0001" ~ "_9999"
            and name[-4:].isdigit()  # ensure the suffix is numeric
        ])

        first_image_name = image_var_name + "_0001"

        
        def parse_single_image():
            stack_data = loadmat(p, variable_names=corresponding_image_names, squeeze_me=True, 
                                  struct_as_record=False)
            for image_number in corresponding_image_names:
                image_array = stack_data[image_number]
                image_idx = int(var_name[2:])
                img_file_name = file_name.replace(".mes", f"_img{image_idx:03d}_{image_number[-4:]}")
                save_path = os.path.join(dir_name, "single_image", "IMG_" + img_file_name + ".tif")
                save_multipage_tiff(image_array[:, :, None].transpose(1, 0, 2)[::-1], save_path)  # add a dummy Z dimension


        def parse_zstack():
            zstack_idx = int(var_name[2:])
            num_stacks = len(corresponding_image_names)
            print(f"Found {num_stacks} stacks for {var_name}, "
                  f"ranging from {corresponding_image_names[0][-4:]} to {corresponding_image_names[-1][-4:]}")
            assert num_stacks == int(corresponding_image_names[-1][-4:]), \
                f"Unmatched number of stacks for {var_name}: expected {int(corresponding_image_names[-1][-4:])}, found {num_stacks}"
            
            stack_data = loadmat(p, variable_names=corresponding_image_names, squeeze_me=True, 
                                  struct_as_record=False)
            stack_frames = [stack_data[frame_name] for frame_name in corresponding_image_names]
            stack_array = np.stack(stack_frames, axis=0).transpose(0, 2, 1)[:, ::-1]  # ZYX

            Df = loadmat(p, variable_names=[var_name], squeeze_me=True, 
                         struct_as_record=False)[var_name]
            FirstDf = Df[0]
            zstack_info = {
                "WidthPixelNum": FirstDf.Width,
                "WidthStep": FirstDf.WidthStep,
                "HeightPixelNum": FirstDf.Height,
                "HeightStep": FirstDf.HeightStep,
                "DepthPixelNum": FirstDf.D3Size,
                "DepthStep": FirstDf.D3Step,
                "AverageFrameNum": FirstDf.Average,
                "FrameDepths": [single_Df.Zlevel for single_Df in Df]        
            }
            
            zstack_file_name = file_name.replace(".mes", f"_ZStacks{zstack_idx:03d}")
            save_path = os.path.join(dir_name, "ZSTACK_" + zstack_file_name + ".ome.tif")
            
            save_zstack_tiff(stack_array, zstack_info, save_path)

            for k, v in zstack_info.items():
                zstack_info[k] = pd.Series(v)
            save_path = os.path.join(dir_name, "ZSTACKINFO_" + zstack_file_name + ".xlsx")
            write_normal_dataframe(pd.DataFrame(zstack_info), f"ZStacks{zstack_idx:03d}", save_path)

        def parse_recording(current_image_name):
            # recording index
            recording_idx = int(var_name[2:])
            print(f"Found recording {recording_idx} with shape {shapes[current_image_name]}, start parsing...")
            image_sub_names = current_image_name.split("_")
            assert image_sub_names[0] == "If" + var_name[2:] or image_sub_names[0] == "IF" + var_name[2:], \
                f"Image variable name {current_image_name} does not match expected pattern 'If{recording_idx:03d}_xxxx'"
            image_idx = int(image_sub_names[1])

            recording_file_name = file_name.replace(".mes", f"_Recording{recording_idx:03d}_{image_idx:04d}")
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

            image_data = loadmat(p, variable_names=[current_image_name], squeeze_me=True, 
                            struct_as_record=False)[current_image_name]
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


        if len(corresponding_image_names) > 100:  # likely a zstack
            print("-" * 4 + f"Z-stack!")
            parse_zstack()
        elif first_image_name in shapes and shapes[first_image_name][1] >= shapes[first_image_name][0] * 10:  # likely a recording
            print("-" * 4 + f"Recording!")
            for tmp_image_name in corresponding_image_names:
                if shapes[tmp_image_name][1] >= shapes[tmp_image_name][0] * 10:
                    parse_recording(tmp_image_name)
        else:
            print("-" * 4 + f"Single image!")
            parse_single_image()
        
        print("-" * 32)


def parse_all_mes_under_dir(dir_path: str):
    all_mes_files = search_pattern_file("*.mes", dir_path)
    print(f"Found {len(all_mes_files)} .mes files under {dir_path}, start parsing...")
    for p in Path(dir_path).rglob("*.mes"):
        parse_mes(p)
    print("Parsing complete!")
    

if __name__ == "__main__":
    parse_all_mes_under_dir(r"C:\Users\maxyc\PycharmProjects\Ratatouille\ingredients\PassivePuff_HighFreqImaging\HighFreqImaging_202607\M022_TFR_10M")
    # parse_all_mes_under_dir(r"C:\Users\maxyc\PycharmProjects\Ratatouille\ingredients\HeadFixedTraining_CalciumImaging\NewMice")