import os
import os.path as path
import time
from glob import glob
import subprocess
from typing import List
from tqdm import tqdm
from send2trash import send2trash

from kitchen.configs import routing
from kitchen.structure.hierarchical_data_structure import Cohort, DataSet
from kitchen.video.video_settings import INPUT_VIDEO_FPS, OUTPUT_VIDEO_FPS, SUPPORT_VIDEO_FORMAT


def find_all_video_path(dir_path: str, format: str) -> List[str]:
    """Find all video file under dir_path with specific format."""
    assert format[0] == ".", f"format should start with ., but got '{format}'"
    assert format in SUPPORT_VIDEO_FORMAT, f"format {format} not supported, should be one of {SUPPORT_VIDEO_FORMAT}"
    all_video_path = []
    for dir_path, _ , _ in os.walk(dir_path):
        pattern = os.path.join(dir_path, '**', 'video', f'*{format}')
        all_video_path.extend(glob(pattern, recursive=True))
    all_video_path = list(set(all_video_path))
    return all_video_path


def video_convert(dir_path: str, src_format: str = ".h264", dst_format: str = ".avi") -> None:
    """Convert all video file under dir_path from src_format to dst_format."""
    assert dst_format[0] == ".", f"dst_format should start with ., but got '{dst_format}'"
    assert dst_format in SUPPORT_VIDEO_FORMAT, f"dst_format {dst_format} not supported, should be one of {SUPPORT_VIDEO_FORMAT}"

    print(f"Converting all {src_format} file under {dir_path} to {dst_format}...")
    all_video_path = find_all_video_path(dir_path, src_format)

    # convert all video file under dir_path to dst_format
    for file_path in tqdm(all_video_path, desc="Converting ", unit="video"):
        tmp_dir, tmp_file = path.dirname(file_path), path.basename(file_path)
        output_file = tmp_file.replace(src_format, dst_format)
        output_path = path.join(tmp_dir, output_file)
        if path.exists(output_path):
            continue
        
        command = (r"ffmpeg -framerate {} -i {} -q:v 6 -vf fps={} "
                r"-hide_banner -loglevel warning {}").format(
                    INPUT_VIDEO_FPS, file_path, OUTPUT_VIDEO_FPS, output_path)
        
        print(command)
        time_start = time.time()
        try:
            subprocess.run(command, check=True, shell=True)
            print(f"Conversion successful: {output_path}, takes {time.time()-time_start:.2f}s")
            """Delete the original file"""
            send2trash(file_path)
        except Exception as e:
            raise ValueError(f"Unknown error occurred: {e}")


def dataset_interface_h264_2_avi(data_set: DataSet):
    """Ask for whether to convert video format"""
    convert_flag = input("Convert video format from h264 to avi? ([y]/n): ")
    convert_flag = True if convert_flag == "y" else False
    if convert_flag:
        for cohort_node in data_set.select("cohort"):
            assert isinstance(cohort_node, Cohort)
            video_convert(dir_path=routing.default_data_path(cohort_node), src_format=".h264", dst_format=".avi")