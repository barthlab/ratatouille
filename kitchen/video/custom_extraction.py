from functools import cached_property
from typing import Tuple
import numpy as np
import cv2
import os
import os.path as path
import pandas as pd
from tqdm import tqdm

from kitchen.configs import routing
from kitchen.settings.behavior import OPTICAL_FLOW_EXTRACTED_BEHAVIOR_TYPES, VIDEO_EXTRACTED_BEHAVIOR_TYPES
from kitchen.structure.hierarchical_data_structure import Cohort, DataSet
from kitchen.utils.sequence_kit import find_only_one
from kitchen.video.format_converter import find_all_video_path
from kitchen.video.video_settings import CUSTOM_EXTRACTION_VIDEO_FORMAT, CUSTOM_EXTRACTION_PREFIX


def OneFrame(capture: cv2.VideoCapture) -> Tuple[bool, np.ndarray]:
    """Read and convert single frame to grayscale."""
    ret0, frame = capture.read()
    if ret0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return ret0, frame


def AvgMultipleFrames(capture: cv2.VideoCapture, frame_num: int =10) -> np.ndarray:
    """Average multiple frames for stable background reference.

    Args:
        capture: Video capture object
        frame_num: Number of frames to average (default: 10)

    Returns:
        Averaged frame as uint8 array

    Raises:
        EOFError: If video ends before collecting required frames
    """
    frame_list = []
    for i in range(frame_num):
        ret0, frame = OneFrame(capture)
        if ret0:
            frame_list.append(frame.astype(np.float32))
        else:
            raise EOFError("Video ended before expected.")
    return np.mean(frame_list, axis=0).astype(np.uint8)


class BodyPartExtractor:
    """Interactive GUI for extracting body part motion from video using optical flow.

    Processing pipeline:
    1. Validate video format and body part type
    2. Generate background reference frame
    3. Interactive ROI selection with mouse drawing
    4. Optical flow calculation on selected region
    5. Save motion intensity data to CSV

    Args:
        video_path: Path to input video file
        part_name: Body part to extract (must be in VIDEO_EXTRACTED_BEHAVIOR_TYPES)
        overwrite: Whether to overwrite existing ROI selection
    """
    def __init__(self, video_path: str, part_name: str, overwrite=False):
        self.video_path = video_path
        self.part_name = part_name.upper()
        self.format_checking()

        self.display_sample = AvgMultipleFrames(self.get_cap(), frame_num=10)
        self.points = []
        self.bbox, self.mask = (0, 0, 0, 0), np.zeros(self.display_sample.shape, dtype=np.uint8)
        self.loaded = self.select_roi(overwrite=overwrite)

    def format_checking(self):
        """Validate video file format and body part type."""
        dir_name, file_name = path.split(self.video_path)
        assert path.exists(self.video_path), f"Video {self.video_path} does not exist."
        assert file_name.endswith(CUSTOM_EXTRACTION_VIDEO_FORMAT),  \
            f"Only support {CUSTOM_EXTRACTION_VIDEO_FORMAT} format, but got video: {file_name}."
        assert self.part_name.upper() in VIDEO_EXTRACTED_BEHAVIOR_TYPES, \
            f"Only support {VIDEO_EXTRACTED_BEHAVIOR_TYPES} extraction, but got {self.part_name}."
        assert file_name.startswith(CUSTOM_EXTRACTION_PREFIX), \
            f"Only support {CUSTOM_EXTRACTION_PREFIX} prefix, but got video: {file_name}."

    @cached_property
    def session_name(self) -> str:
        """Extract session name from video filename."""
        dir_name, file_name = path.split(self.video_path)
        matched_prefix = find_only_one(CUSTOM_EXTRACTION_PREFIX, _self = lambda x: file_name.startswith(x))
        matched_format = find_only_one(CUSTOM_EXTRACTION_VIDEO_FORMAT, _self = lambda x: file_name.endswith(x))
        session_name = file_name[len(matched_prefix): -len(matched_format)]
        print(session_name)
        return session_name

    @cached_property
    def result_save_path(self) -> str:
        """Path for saving extracted motion data CSV."""
        dir_name, file_name = path.split(self.video_path)
        save_path = path.join(dir_name, "..", self.part_name.lower(), f"{self.part_name}_{self.session_name}.csv")
        os.makedirs(path.dirname(save_path), exist_ok=True)
        return save_path

    @cached_property
    def archive_path(self) -> str:
        """Path for saving ROI selection archive."""
        dir_name, file_name = path.split(self.video_path)
        archive_path = path.join(dir_name, f"{self.part_name}_{self.session_name}.npz")
        return archive_path

    def get_cap(self) -> cv2.VideoCapture:
        """Create new video capture object."""
        return cv2.VideoCapture(self.video_path)

    def select_roi(self, overwrite) -> bool:
        """Interactive ROI selection with mouse drawing or load from archive.

        Controls:
        - Mouse drag to draw polygon ROI
        - 's' to save selection
        - 'n' to skip without saving
        - 'q' to quit
        """
        cv2.namedWindow(self.part_name)
        cv2.moveWindow(self.part_name, 0, 0)
        if path.exists(self.archive_path) and (not overwrite):
            archive_dict = np.load(self.archive_path)
            self.bbox, self.mask, self.points = archive_dict['bbox'], archive_dict['mask'], archive_dict['points']
            self.display_selected_region()
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
            return True
        else:
            new_window_for_selection = "Select ROI for " + self.part_name
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
                    np.savez(self.archive_path, bbox=self.bbox, mask=self.mask, points=self.points)
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
        """Handle mouse events for polygon ROI drawing."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points = [(x, y), ]
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            self.points.append((x, y))
            self.refresh_display()
        elif event == cv2.EVENT_LBUTTONUP:
            self.points.append((x, y))
            self.refresh_display()
            self.display_selected_region()

    def refresh_display(self):
        """Update display with current polygon drawing."""
        frame_name = "Select ROI for " + self.part_name
        tmp_img = self.display_sample.copy()
        for i in range(len(self.points)-1):
            cv2.line(tmp_img, self.points[i], self.points[i + 1], (0, 255, 0), 2)
            cv2.line(tmp_img, self.points[i], self.points[i + 1], (255, 255, 255), 1)
        cv2.line(tmp_img, self.points[-1], self.points[0], (0, 255, 0), 2)
        cv2.line(tmp_img, self.points[-1], self.points[0], (255, 255, 255), 1)
        cv2.imshow(frame_name, tmp_img)

    def display_selected_region(self):
        """Show final selected region with mask applied."""
        x, y, w, h = cv2.boundingRect(np.array(self.points))
        mask = np.zeros(self.display_sample.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(self.points)], (255, 0, 0))
        mask = mask[y:y+h, x:x+w]
        selected_rec = self.display_sample.copy()[y:y+h, x:x+w]
        selected_rec[mask == 0] = 0
        self.bbox, self.mask = (x, y, w, h), mask
        cv2.imshow(self.part_name, selected_rec)

    def optical_flow_extraction(self):
        """Extract motion intensity using Farneback optical flow algorithm.

        Processing pipeline:
        1. Crop and mask frames to selected ROI
        2. Calculate dense optical flow between consecutive frames
        3. Compute motion magnitude and average across ROI
        4. Save frame-by-frame intensity data to CSV

        Displays real-time visualization of flow magnitude and processed frames.
        """
        if not self.loaded:
            print(f"ROI not selected for {self.part_name}, skip extraction.")
            return
        
        cv2.namedWindow("Optical Flow")
        x, y, w, h = self.bbox
        mask = self.mask

        def preprocess(tmp_frame):
            """Preprocess frame for motion extraction."""
            crop_frame = tmp_frame[y:y+h, x:x+w]
            crop_frame[mask == 0] = 0
            return cv2.resize(crop_frame, dsize=None, fx=1, fy=1)

        cap = self.get_cap()
        frame_cnt = 0
        _, pre_frame = OneFrame(cap)
        pre_frame = preprocess(pre_frame)
        intensity_list = []

        # Get total frame count for tqdm
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
            while True:
                ret0, frame = OneFrame(cap)
                if not ret0:
                    break
                frame_cnt += 1

                cur_frame = preprocess(frame)
                flow = cv2.calcOpticalFlowFarneback(
                    prev=pre_frame, next=cur_frame, flow=np.array([]), pyr_scale=0.5, levels=3, winsize=5,
                    iterations=3, poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mag = np.asarray(mag, dtype=np.float32)
                intensity_list.append(np.mean(mag))
                pre_frame = cur_frame

                # Update progress bar with current intensity
                pbar.set_postfix({'Intensity': f'{np.mean(mag):.3f}'})
                pbar.update(1)

                # Visualization
                mag_mask = np.clip(mag*5, a_min=0, a_max=255).astype(np.uint8)
                display_vis = np.concatenate([mag_mask, cur_frame], axis=1)
                cv2.imshow('Optical Flow', display_vis)
                cv2.waitKey(1)
        cv2.destroyAllWindows()
        save_dict = pd.DataFrame({self.part_name.lower(): intensity_list})
        save_dict.to_csv(self.result_save_path, index_label="Frame")


def default_collection(data_set: DataSet, overwrite=False):
    """Preload all body parts for motion extraction."""
    all_body_parts = []
    for cohort_node in data_set.select("cohort"):
        assert isinstance(cohort_node, Cohort)
        all_video_path = find_all_video_path(routing.default_data_path(cohort_node), ".avi")
        for video_path in all_video_path:
            for body_part in OPTICAL_FLOW_EXTRACTED_BEHAVIOR_TYPES:
                all_body_parts.append(BodyPartExtractor(video_path, body_part, overwrite=overwrite))
    """Extract motion intensity for all body parts"""
    for body_part in all_body_parts:
        body_part.optical_flow_extraction()