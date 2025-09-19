
import os
import os.path as path
import subprocess
import sys
import numpy as np
import pandas as pd

from kitchen.configs import routing
from kitchen.structure.hierarchical_data_structure import Cohort, DataSet
from kitchen.utils.sequence_kit import find_only_one
from kitchen.media.format_converter import find_all_video_path
from kitchen.media.video_settings import CUSTOM_EXTRACTION_PREFIX, CUSTOM_EXTRACTION_VIDEO_FORMAT

def run_facemap():
    """
    Runs the 'python -m facemap' command and waits for the process to exit.

    This function uses subprocess.run() which executes the command and blocks
    until the command completes.
    """
    command = [sys.executable, "-m", "facemap"]
    
    print("--- Starting Facemap ---")
    print(f"Running command: {' '.join(command)}")
    print("Your script will wait here until you close the Facemap GUI...")

    try:
        result = subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True
        )

        print("\n--- Facemap has been closed ---")
        print("Facemap exited successfully.")
        
        if result.stdout:
            print("\n--- Facemap Standard Output: ---")
            print(result.stdout)
        if result.stderr:
            print("\n--- Facemap Standard Error: ---")
            print(result.stderr)

    except FileNotFoundError:
        print("\n--- ERROR ---")
        print(f"Error: The command '{sys.executable}' was not found.")
        print("Please ensure Python is in your system's PATH.")

    except subprocess.CalledProcessError as e:
        # This block runs if facemap exits with an error code.
        print("\n--- ERROR ---")
        print("Facemap exited with an error.")
        print(f"Return code: {e.returncode}")
        print("\n--- Standard Output: ---")
        print(e.stdout)
        print("\n--- Standard Error: ---")
        print(e.stderr)
        
    except Exception as e:
        # Catch any other unexpected errors
        print(f"\nAn unexpected error occurred: {e}")


def pupil_save_path(video_path: str) -> str:
    dir_name, file_name = path.split(video_path)
    matched_prefix = find_only_one(CUSTOM_EXTRACTION_PREFIX, _self = lambda x: file_name.startswith(x))
    matched_format = find_only_one(CUSTOM_EXTRACTION_VIDEO_FORMAT, _self = lambda x: file_name.endswith(x))
    session_name = file_name[len(matched_prefix): -len(matched_format)]
    save_path = path.join(dir_name, "..", 'pupil', f"PUPIL_{session_name}.csv")
    os.makedirs(path.dirname(save_path), exist_ok=True)
    return save_path


def default_collection(data_set: DataSet):
    """Ask for whether to open Facemap for pupil extraction."""
    open_flag = input("Open Facemap for pupil extraction? ([y]/n): ")
    open_flag = True if open_flag == "y" else False
    if open_flag:
        run_facemap()

    """Move facemap processed result to PUPIL folder"""
    for cohort_node in data_set.select("cohort"):
        assert isinstance(cohort_node, Cohort)
        all_video_path = find_all_video_path(routing.default_data_path(cohort_node), ".avi")
        for video_path in all_video_path:
            processed_path = video_path.replace(".avi", "_proc.npy")
            to_saved_path = pupil_save_path(video_path)
            if path.exists(processed_path):
                print(f"Processing {processed_path}")
                tmp_data = np.load(processed_path, allow_pickle=True).item()['pupil'][0]
                area_smooth = tmp_data['area_smooth']
                save_dict = pd.DataFrame({"Pupil": area_smooth})
                save_dict.to_csv(to_saved_path, index_label="Frame")
                print(f"Saved to {to_saved_path}")
