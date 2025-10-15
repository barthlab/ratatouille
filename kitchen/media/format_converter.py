import os
import os.path as path
import time
from glob import glob
import subprocess
from typing import List
import numpy as np
from tqdm import tqdm
from send2trash import send2trash
import logging

from kitchen.configs import routing
from kitchen.settings.loaders import DATA_HODGEPODGE_MODE
from kitchen.structure.hierarchical_data_structure import Cohort, DataSet
from kitchen.media.video_settings import INPUT_VIDEO_FPS, OUTPUT_VIDEO_FPS, SUPPORT_VIDEO_FORMAT, TIFF_STACK_FPS, TIFF_STACK_CRF

logger = logging.getLogger(__name__)

def find_all_video_path(dir_path: str, format: str) -> List[str]:
    """Find all video file under dir_path with specific format."""
    assert format[0] == ".", f"format should start with ., but got '{format}'"
    assert format in SUPPORT_VIDEO_FORMAT, f"format {format} not supported, should be one of {SUPPORT_VIDEO_FORMAT}"
    if DATA_HODGEPODGE_MODE:
        pattern = f'*{format}'
    else:
        pattern = os.path.join('video', f'*{format}')
    all_video_path = routing.search_pattern_file(pattern, dir_path)
    return all_video_path


def video_convert(dir_path: str, src_format: str = ".h264", dst_format: str = ".avi") -> None:
    """Convert all video file under dir_path from src_format to dst_format."""
    assert dst_format[0] == ".", f"dst_format should start with ., but got '{dst_format}'"
    assert dst_format in SUPPORT_VIDEO_FORMAT, f"dst_format {dst_format} not supported, should be one of {SUPPORT_VIDEO_FORMAT}"

    logger.info(f"Converting all {src_format} file under {dir_path} to {dst_format}...")
    all_video_path = find_all_video_path(dir_path, src_format)
    logger.info(f"Found {len(all_video_path)} video file to convert")
    
    # convert all video file under dir_path to dst_format
    for file_path in tqdm(all_video_path, desc="Converting ", unit="video"):
        tmp_dir, tmp_file = path.dirname(file_path), path.basename(file_path)
        output_file = tmp_file.replace(src_format, dst_format)
        output_path = path.join(tmp_dir, output_file)
        if path.exists(output_path):
            logger.info(f"Video {output_path} already exists, skipping...")
            continue
        
        command = (r"ffmpeg -framerate {} -i {} -q:v 6 -vf fps={} "
                r"-hide_banner -loglevel warning {}").format(
                    INPUT_VIDEO_FPS, file_path, OUTPUT_VIDEO_FPS, output_path)
        
        logger.info(command)
        time_start = time.time()
        try:
            subprocess.run(command, check=True, shell=True)
            logger.info(f"Conversion successful: {output_path}, takes {time.time()-time_start:.2f}s")
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


def stack_tiff_to_video(dir_path: str):
    """
    Convert TIFF stacks in subdirectories to MP4 videos.
    
    Enumerates all folders under dir_path, finds TIFF files matching the Basler_*.tiff pattern
    in each folder, sorts them numerically, and converts them to MP4 videos using ffmpeg.
    Output videos are saved as {folder_name}.mp4 in the parent directory.
    """
    print(f"Converting TIFF stacks to videos in {dir_path}...")
    
    # Find all subdirectories
    subdirs = [d for d in os.listdir(dir_path) if path.isdir(path.join(dir_path, d))]
    
    if not subdirs:
        print("No subdirectories found.")
        return
    
    for folder_name in tqdm(subdirs, desc="Processing folders", unit="folder"):
        folder_path = os.path.join(dir_path, folder_name)
        output_video = os.path.join(dir_path, f"VIDEO_{folder_name}.mp4")
        
        # Skip if video already exists
        if path.exists(output_video):
            print(f"Video {output_video} already exists, skipping...")
            continue
        
        # Find TIFF files with Basler pattern
        tiff_files = routing.search_pattern_file('Basler_*.tiff', folder_path)
        
        if not tiff_files:
            print(f"No TIFF files found in {folder_path}, skipping...")
            continue

        # Sort files numerically by the frame number
        try:
            sorted_files = sorted(
                tiff_files,
                key=lambda f: int(path.splitext(path.basename(f))[0].split('_')[-1])
            )
            print(f"Found and sorted {len(sorted_files)} TIFF files in {folder_name}")
            frame_diffs = np.diff([int(path.splitext(path.basename(f))[0].split('_')[-1]) for f in sorted_files])
        except (ValueError, IndexError):
            print(f"Error: Could not parse frame numbers from filenames in {folder_path}")
            continue
        
        # Create temporary file list for FFMPEG
        temp_filelist = path.join(folder_path, 'filelist.txt')
        
        with open(temp_filelist, 'w', encoding='ascii') as f:
            for filename, frame_diff in zip(sorted_files, frame_diffs):
                # Use absolute path and escape for FFMPEG
                abs_path = path.abspath(filename).replace('\\', '/')
                f.write(f"file '{abs_path}'\nduration {frame_diff/TIFF_STACK_FPS:.6f}\n")
        
        # Build FFMPEG command using string format like video_convert
        command = (r"ffmpeg -y -f concat -safe 0 -i {} -c:v libx264 "
                  r"-pix_fmt yuv420p -crf {} -r {} -hide_banner -loglevel warning {}").format(
                      temp_filelist, TIFF_STACK_CRF, TIFF_STACK_FPS, output_video)
        
        print(f"Creating video: {output_video}")
        print(command)
        time_start = time.time()
        try:
            subprocess.run(command, check=True, shell=True)
            print(f"Conversion successful: {output_video}, takes {time.time()-time_start:.2f}s")
        except Exception as e:
            print(f"Error during FFMPEG execution for {folder_name}: {e}")
        finally:
            # Clean up temporary file
            if path.exists(temp_filelist):
                os.remove(temp_filelist)