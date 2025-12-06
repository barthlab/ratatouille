import os
import os.path as path
import subprocess
import sys
from functools import cached_property
from typing import Optional, Tuple

import cv2
import numpy as np

from kitchen.configs import routing
from kitchen.structure.hierarchical_data_structure import Cohort, DataSet
from kitchen.utils.sequence_kit import find_only_one
from kitchen.media.format_converter import find_all_video_path
from kitchen.media.video_settings import CUSTOM_EXTRACTION_PREFIX, CUSTOM_EXTRACTION_VIDEO_FORMAT
from kitchen.media.custom_extraction import OneFrame, AvgMultipleFrames, video_format_checking


class EyeRoiSelector:
    """Rectangle ROI selector for eye roi extraction."""

    def __init__(self, video_path: str, overwrite: bool = False):
        self.video_path = video_path
        self.overwrite = overwrite

        self.display_sample = AvgMultipleFrames(self.get_cap(), frame_num=10)
        self.bbox: Optional[Tuple[int, int, int, int]] = None
        self.loaded = self.select_roi(overwrite=overwrite)
    
    @cached_property
    def archive_path(self) -> str:
        dir_name, file_name = path.split(self.video_path)
        matched_prefix = find_only_one(CUSTOM_EXTRACTION_PREFIX, _self=lambda x: file_name.startswith(x))
        matched_format = find_only_one(CUSTOM_EXTRACTION_VIDEO_FORMAT, _self=lambda x: file_name.endswith(x))
        session_name = file_name[len(matched_prefix): -len(matched_format)]
        return path.join(dir_name, f"EYE_{session_name}.npz")

    def get_cap(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(self.video_path)
    
    def select_roi(self, overwrite: bool) -> bool:
        cv2.namedWindow("Eye")
        cv2.moveWindow("Eye", 0, 0)
        if path.exists(self.archive_path) and (not overwrite):
            try:
                archive_dict = np.load(self.archive_path)
                x, y, w, h = archive_dict["bbox"].tolist()
                self.bbox = (int(x), int(y), int(w), int(h))
                self.display_selected_region()
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Cannot load archive {self.archive_path}: {e}")
                print("Redo ROI selection.")
                self.select_roi(overwrite=True)
            return True
        else:
            new_window_for_selection = "Select Eye ROI"
            cv2.namedWindow(new_window_for_selection)
            cv2.setMouseCallback(new_window_for_selection, self.mouse_callback)
            cv2.imshow(new_window_for_selection, self.display_sample)
            while True:
                choice = cv2.waitKey(0)
                if choice & 0xFF == ord("q"):
                    print("Quit!")
                    cv2.destroyAllWindows()
                    exit()
                elif choice & 0xFF == ord("s"):
                    np.savez(self.archive_path, bbox=self.bbox)
                    print(f"{self.archive_path} Saved!")                    
                    cv2.destroyAllWindows()
                    return True
                elif choice & 0xFF == ord("n"):
                    print("Skipped!")
                    cv2.destroyAllWindows()
                    return False
                else:
                    print(f"Invalid input {choice}, please try again.")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.bbox = (x, y, 0, 0)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            min_length = min(x - self.bbox[0], y - self.bbox[1])
            self.bbox = (self.bbox[0], self.bbox[1], min_length, min_length)
            self.refresh_display()
        elif event == cv2.EVENT_LBUTTONUP:
            min_length = min(x - self.bbox[0], y - self.bbox[1])
            self.bbox = (self.bbox[0], self.bbox[1], min_length, min_length)
            self.refresh_display()
            self.display_selected_region()

    def refresh_display(self):
        """Update display with current rectangle drawing."""
        frame_name = "Select Eye ROI"
        tmp_img = self.display_sample.copy()
        if self.bbox:
            x, y, w, h = self.bbox
            cv2.rectangle(tmp_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow(frame_name, tmp_img)

    def display_selected_region(self):
        """Show final selected region."""
        x, y, w, h = self.bbox
        selected_rec = self.display_sample.copy()[y:y+h, x:x+w]
        cv2.imshow("Eye", selected_rec)
         

def default_collection(data_set: DataSet, _expected_format: str = ".mp4", overwrite=False, reextract=False):
    """Collect pupil ROI via GUI and run mEye predictor."""

    for cohort_node in data_set.select("cohort"):
        assert isinstance(cohort_node, Cohort)
        all_video_path = find_all_video_path(routing.default_data_path(cohort_node), _expected_format)
        for video_path in all_video_path:
            try:
                video_format_checking(video_path)
            except AssertionError as e:
                print(f"Skip {video_path} due to: {e}")
                continue

            EyeRoiSelector(video_path, overwrite=overwrite)

