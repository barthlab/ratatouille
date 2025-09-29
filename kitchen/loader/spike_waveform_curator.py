"""
Spike Waveform Curation Tool

Interactive GUI application for manually curating spike waveforms from electrophysiological recordings.
Provides UMAP-based visualization of spike waveforms with elliptical selection tools for quality control.

The tool allows users to:
- Visualize spike waveforms in UMAP space (both raw and z-scored)
- Select/deselect spikes using interactive ellipses
- View waveform plots for selected/unselected spikes
- Save and load curation selections
- Apply final selections to filter spike data

Main workflow:
1. Load spike waveforms from a Node's potential data
2. Generate UMAP embeddings for visualization
3. Allow manual selection using elliptical regions
4. Save selection mask and apply to original spike data
"""

# Standard library imports
import logging
import os

import os.path
import pickle
import sys
from typing import Optional

# Third-party imports
import numpy as np
from kitchen.utils.numpy_kit import zscore
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtCore import QPointF, QCoreApplication, QEventLoop
from PyQt5.QtGui import QTransform
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QPushButton, QMessageBox)
from sklearn.cluster import KMeans
from umap import UMAP

# Local imports
from kitchen.configs import routing
from kitchen.operator.grouping import grouping_timeseries
from kitchen.settings.potential import CURATION_SPIKE_RANGE_RELATIVE_TO_ALIGNMENT, SPIKE_RANGE_RELATIVE_TO_ALIGNMENT
from kitchen.settings.timeline import TRIAL_ALIGN_EVENT_DEFAULT
from kitchen.structure.hierarchical_data_structure import Node

pg.setConfigOptions(useOpenGL=True)

logger = logging.getLogger(__name__)

# --- Singleton instances for the GUI ---
_app_instance: Optional[QApplication] = None
_gui_instance: Optional['MainWindow'] = None


class SelectionEllipse:
    """
    Represents an elliptical selection region in 2D space.

    Used for selecting/deselecting points in UMAP visualization. Supports
    rotation, resizing, and translation. Uses opt-out logic where points
    inside ellipses are excluded from the final selection.

    Attributes:
        center: Center point of the ellipse
        width: Width of the ellipse
        height: Height of the ellipse
        rotation: Rotation angle in degrees
        selected_indices: List of point indices inside this ellipse
    """
    def __init__(self):
        """Initialize ellipse with default parameters."""
        self.center = QPointF(0, 0)
        self.width = 30
        self.height = 30
        self.rotation = 0.0
        self.selected_indices = []

    def contains_point(self, point):
        """
        Check if a point is inside this ellipse.

        Args:
            point: QPointF representing the point to test

        Returns:
            bool: True if point is inside the ellipse, False otherwise
        """
        transform = QTransform()
        transform.translate(self.center.x(), self.center.y())
        transform.rotate(self.rotation)
        transformed_point, ok = transform.inverted()
        if not ok:
            return False

        transformed_point = transformed_point.map(point)

        if self.width == 0 or self.height == 0:
            return False
        normalized_x = transformed_point.x() / (self.width / 2)
        normalized_y = transformed_point.y() / (self.height / 2)
        return (normalized_x ** 2 + normalized_y ** 2) <= 1

    def update_selection(self, points):
        """
        Update the list of point indices that fall inside this ellipse.

        Args:
            points: Array of 2D points to test against the ellipse

        Returns:
            list: Indices of points that fall inside the ellipse
        """
        self.selected_indices = []
        transform = QTransform()
        transform.translate(self.center.x(), self.center.y())
        transform.rotate(self.rotation)
        inv, ok = transform.inverted()
        if not ok:
            return self.selected_indices

        half_width = self.width / 2
        half_height = self.height / 2
        if half_width == 0 or half_height == 0:
            return self.selected_indices

        for i, point in enumerate(points):
            transformed_point = inv.map(QPointF(point[0], point[1]))
            normalized_x = transformed_point.x() / half_width
            normalized_y = transformed_point.y() / half_height
            if normalized_x ** 2 + normalized_y ** 2 <= 1:
                self.selected_indices.append(i)
        return self.selected_indices


class UMAPPlotWidget(pg.PlotWidget):
    """
    Interactive UMAP visualization widget for spike waveform data.

    Displays spike waveforms as points in 2D UMAP space with interactive
    elliptical selection tools. Supports multiple ellipses for complex
    selections using opt-out logic (points inside ellipses are excluded).

    Features:
    - Interactive ellipse creation, resizing, rotation, and movement
    - Real-time selection updates
    - Visual feedback with color-coded points
    - Manual point selection support
    """

    def __init__(self, parent=None):
        """
        Initialize the UMAP plot widget.

        Args:
            parent: Parent widget (typically MainWindow)
        """
        super().__init__(parent)
        self.parent = parent
        self.setBackground('w')
        self.points = None

        self.ellipses = []
        self.manual_selection = set()

        self.active_ellipse_index = None
        self.ellipse_items = []

        self.scatter = None
        self.resize_handle_item = None
        self.rotate_handle_item = None
        self.drag_start = None
        self.is_adjusting_ellipse = False
        self.initial_aspect_ratio = None

        self.getPlotItem().setLabel('bottom', 'UMAP 1')
        self.getPlotItem().setLabel('left', 'UMAP 2')
        self.getPlotItem().showGrid(x=True, y=True)

    def set_data(self, umap_coordinates, cluster_labels, ellipses, manual_selection, create_default_ellipse=False):
        """
        Set the UMAP data and initialize the visualization.

        Args:
            umap_coordinates: 2D array of UMAP coordinates for each spike
            cluster_labels: Cluster labels from K-means clustering
            ellipses: List of existing SelectionEllipse objects
            manual_selection: Set of manually selected point indices
            create_default_ellipse: Whether to create a default ellipse if none exist
        """
        self.clear()
        self.points = umap_coordinates
        self.labels = cluster_labels

        self.ellipses = ellipses
        self.manual_selection = manual_selection

        self.brushes = [pg.mkBrush(100, 100, 100, 150) for _ in range(len(umap_coordinates))]

        spots = [{'pos': umap_coordinates[i], 'data': i} for i in range(len(umap_coordinates))]
        self.scatter = pg.ScatterPlotItem(
            spots=spots,
            size=10,
            brush=self.brushes,
            pen=pg.mkPen(None),
            symbol='o'
        )
        self.addItem(self.scatter)

        if not self.ellipses and create_default_ellipse:
            initial_ellipse = SelectionEllipse()
            target_label = 0
            cluster_indices = np.where(self.labels == target_label)[0]

            points_for_ellipse = self.points[cluster_indices] if len(cluster_indices) > 0 else self.points

            if len(points_for_ellipse) > 0:
                center_x = np.mean(points_for_ellipse[:, 0])
                center_y = np.mean(points_for_ellipse[:, 1])
                initial_ellipse.center = QPointF(center_x, center_y)

                std_x = np.std(points_for_ellipse[:, 0])
                std_y = np.std(points_for_ellipse[:, 1])
                initial_ellipse.width = std_x * 5
                initial_ellipse.height = std_y * 5

            self.ellipses.append(initial_ellipse)

        self.active_ellipse_index = 0 if self.ellipses else None

        self.draw_ellipses()
        self.update_selection()

    def add_new_ellipse(self):
        new_ellipse = SelectionEllipse()

        view_rect = self.plotItem.vb.viewRect()
        new_ellipse.center = view_rect.center()
        new_ellipse.width = view_rect.width() / 4
        new_ellipse.height = view_rect.height() / 4

        self.ellipses.append(new_ellipse)
        self.active_ellipse_index = len(self.ellipses) - 1

        self.draw_ellipses()
        self.update_selection()

    def draw_ellipses(self):
        for item in self.ellipse_items:
            self.removeItem(item)
        self.ellipse_items.clear()

        for i, ellipse in enumerate(self.ellipses):
            pen = pg.mkPen(color=(0, 0, 255), width=2) if i == self.active_ellipse_index else pg.mkPen(color=(0, 100, 255), width=1)

            path_points = []
            cos_rot = np.cos(np.radians(ellipse.rotation))
            sin_rot = np.sin(np.radians(ellipse.rotation))
            for j in range(101):
                angle = 2 * np.pi * j / 100
                x = ellipse.width / 2 * np.cos(angle)
                y = ellipse.height / 2 * np.sin(angle)

                rotated_x = ellipse.center.x() + x * cos_rot - y * sin_rot
                rotated_y = ellipse.center.y() + x * sin_rot + y * cos_rot
                path_points.append((rotated_x, rotated_y))

            path_points = np.array(path_points)
            ellipse_item = pg.PlotCurveItem(pen=pen)
            ellipse_item.setData(path_points[:, 0], path_points[:, 1])
            self.addItem(ellipse_item)
            self.ellipse_items.append(ellipse_item)

        self.add_control_handles()

    def add_control_handles(self):
        """Add visual handles for resizing and rotating the active ellipse."""
        if self.resize_handle_item:
            self.removeItem(self.resize_handle_item)
        if self.rotate_handle_item:
            self.removeItem(self.rotate_handle_item)

        if self.active_ellipse_index is None or self.active_ellipse_index >= len(self.ellipses):
            return

        active_ellipse = self.ellipses[self.active_ellipse_index]

        right_point_x = active_ellipse.center.x() + active_ellipse.width / 2 * np.cos(np.radians(active_ellipse.rotation))
        right_point_y = active_ellipse.center.y() + active_ellipse.width / 2 * np.sin(np.radians(active_ellipse.rotation))
        self.resize_handle_pos = np.array([[right_point_x, right_point_y]])
        self.resize_handle_item = pg.ScatterPlotItem(pos=self.resize_handle_pos, size=15, brush=pg.mkBrush(255, 0, 0, 200), pen=pg.mkPen('w'), symbol='s', zValue=10)
        self.addItem(self.resize_handle_item)

        top_point_x = active_ellipse.center.x() + active_ellipse.height / 2 * np.sin(np.radians(active_ellipse.rotation))
        top_point_y = active_ellipse.center.y() - active_ellipse.height / 2 * np.cos(np.radians(active_ellipse.rotation))
        self.rotate_handle_pos = np.array([[top_point_x, top_point_y]])
        self.rotate_handle_item = pg.ScatterPlotItem(pos=self.rotate_handle_pos, size=15, brush=pg.mkBrush(0, 255, 0, 200), pen=pg.mkPen('w'), symbol='o', zValue=10)
        self.addItem(self.rotate_handle_item)

    def update_selection(self):
        if self.points is None:
            return

        points_in_ellipses = set()
        for ellipse in self.ellipses:
            indices = ellipse.update_selection(self.points)
            points_in_ellipses.update(indices)

        # Use opt-out logic: start with all points, remove those in ellipses, add manual selections
        all_indices = set(range(len(self.points)))
        selection = all_indices.difference(points_in_ellipses)
        selection.update(self.manual_selection)

        # Notify parent of selection change for cross-mode synchronization
        if self.parent:
            self.parent.on_selection_changed(list(selection))

    def update_colors(self, final_selection_indices):
        """Updates the color of the scatter plot points based on a provided list of selected indices."""
        if self.points is None or self.scatter is None:
            return

        selection_set = set(final_selection_indices)
        brushes = self.brushes.copy()
        for i in range(len(self.points)):
            # Selected points are red, unselected points use the default gray brush
            brushes[i] = pg.mkBrush(255, 0, 0, 200) if i in selection_set else self.brushes[i]

        self.scatter.setBrush(brushes)

    def mousePressEvent(self, event):
        pos = self.plotItem.vb.mapSceneToView(event.pos())
        self.is_adjusting_ellipse = False

        if self.active_ellipse_index is not None:
            active_ellipse = self.ellipses[self.active_ellipse_index]
            resize_point = QPointF(self.resize_handle_pos[0][0], self.resize_handle_pos[0][1])
            rotate_point = QPointF(self.rotate_handle_pos[0][0], self.rotate_handle_pos[0][1])

            threshold = 20
            screen_pos = self.plotItem.vb.mapViewToScene(pos)
            screen_resize = self.plotItem.vb.mapViewToScene(resize_point)
            screen_rotate = self.plotItem.vb.mapViewToScene(rotate_point)

            resize_dist = (screen_pos - screen_resize).manhattanLength()
            rotate_dist = (screen_pos - screen_rotate).manhattanLength()

            if resize_dist < threshold:
                self.is_adjusting_ellipse = 'resize'
            elif rotate_dist < threshold:
                self.is_adjusting_ellipse = 'rotate'

            if self.is_adjusting_ellipse:
                self.drag_start = pos
                transform = QTransform().rotate(-active_ellipse.rotation)
                self.initial_local = transform.map(pos - active_ellipse.center)
                super().mousePressEvent(event)
                return

        clicked_on_ellipse = False
        for i in range(len(self.ellipses) - 1, -1, -1):
            if self.ellipses[i].contains_point(pos):
                self.active_ellipse_index = i
                self.is_adjusting_ellipse = 'drag'
                self.drag_start = pos
                clicked_on_ellipse = True
                self.draw_ellipses()
                break

        if not clicked_on_ellipse:
            self.active_ellipse_index = None
            self.draw_ellipses()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag_start is None or not self.is_adjusting_ellipse or self.active_ellipse_index is None:
            super().mouseMoveEvent(event)
            return

        active_ellipse = self.ellipses[self.active_ellipse_index]
        pos = self.plotItem.vb.mapSceneToView(event.pos())

        if self.is_adjusting_ellipse == 'drag':
            dx = pos.x() - self.drag_start.x()
            dy = pos.y() - self.drag_start.y()
            active_ellipse.center = QPointF(active_ellipse.center.x() + dx, active_ellipse.center.y() + dy)

        elif self.is_adjusting_ellipse in ['resize', 'rotate']:
            center_to_pos = pos - active_ellipse.center

            transform = QTransform().rotate(-active_ellipse.rotation)
            local_pos = transform.map(center_to_pos)

            if self.is_adjusting_ellipse == 'resize':
                active_ellipse.width = 2 * abs(local_pos.x())

            elif self.is_adjusting_ellipse == 'rotate':
                active_ellipse.height = 2 * abs(local_pos.y())
                angle = np.degrees(np.arctan2(center_to_pos.y(), center_to_pos.x()))
                active_ellipse.rotation = angle + 90

        self.drag_start = pos
        self.draw_ellipses()
        event.accept()

    def mouseReleaseEvent(self, event):
        self.drag_start = None
        self.is_adjusting_ellipse = False
        self.update_selection()
        super().mouseReleaseEvent(event)


class WaveformPlotWidget(pg.PlotWidget):
    """
    Widget for plotting spike waveforms over time.

    Displays individual waveforms as overlaid traces with optional mean waveform.
    Limits the number of displayed waveforms for performance.
    """

    def __init__(self, parent=None, y_label="Amplitude"):
        """
        Initialize the waveform plot widget.

        Args:
            parent: Parent widget
            y_label: Label for the y-axis
        """
        super().__init__(parent)
        self.setBackground('w')
        plot_item = self.getPlotItem()
        plot_item.setLabel('bottom', 'Time (ms)')
        plot_item.setLabel('left', y_label)
        plot_item.showGrid(x=True, y=True)

    def plot_waveforms(self, waveforms, times, pen, mean_waveform=None, mean_pen=None):
        """
        Plot spike waveforms with optional mean overlay.

        Args:
            waveforms: 2D array of waveforms (n_spikes x n_timepoints)
            times: Time points for the waveforms
            pen: Pen style for individual waveforms
            mean_waveform: Optional mean waveform to overlay
            mean_pen: Pen style for mean waveform
        """
        self.clear()
        if waveforms is None or len(waveforms) == 0:
            return

        max_waveforms_to_plot = 1000

        if len(waveforms) > max_waveforms_to_plot:
            indices_to_plot = np.random.choice(np.arange(len(waveforms)), max_waveforms_to_plot, replace=False)
            waveforms_to_plot = waveforms[indices_to_plot]
        else:
            waveforms_to_plot = waveforms

        x_vals = np.tile(np.append(times, np.nan), len(waveforms_to_plot))
        y_vals = np.insert(waveforms_to_plot, waveforms_to_plot.shape[1], np.nan, axis=1).flatten()
        self.plot(x_vals, y_vals, pen=pen)

        if mean_waveform is not None and mean_pen is not None:
            self.plot(times, mean_waveform, pen=mean_pen)


class SplitWaveformWidget(QWidget):
    """
    Widget displaying waveforms split into three categories: selected, focused, and unselected.

    Shows three synchronized plots:
    - Selected: Waveforms that will be kept after curation
    - Focused: Waveforms in the currently active ellipse
    - Unselected: Waveforms that will be excluded
    """

    def __init__(self, data_type_name: str, parent=None):
        """
        Initialize the split waveform widget.

        Args:
            data_type_name: Name of the data type (e.g., "Raw", "Z-scored")
            parent: Parent widget
        """
        super().__init__(parent)

        self.selected_plot = WaveformPlotWidget(y_label=f"{data_type_name} Amplitude")
        self.focused_plot = WaveformPlotWidget(y_label=f"{data_type_name} Amplitude")
        self.unselected_plot = WaveformPlotWidget(y_label=f"{data_type_name} Amplitude")

        self.selected_plot.setTitle(f"Selected {data_type_name} Waveforms")
        self.focused_plot.setTitle(f"Focused Ellipse {data_type_name} Waveforms")
        self.unselected_plot.setTitle(f"Unselected {data_type_name} Waveforms")

        self.focused_plot.setYLink(self.selected_plot)
        self.unselected_plot.setYLink(self.selected_plot)

        layout = QVBoxLayout()
        layout.addWidget(self.selected_plot)
        layout.addWidget(self.focused_plot)
        layout.addWidget(self.unselected_plot)
        self.setLayout(layout)

        self.black_pen = pg.mkPen(color=(0, 0, 0, 150), width=1)
        self.red_pen = pg.mkPen(color=(255, 0, 0, 150), width=1)
        self.blue_pen = pg.mkPen(color=(0, 0, 255, 150), width=1)
        self.green_pen = pg.mkPen(color=(0, 255, 0), width=2)

    def plot_waveforms(self, waveforms, times, selected_indices=None, focused_indices=None):
        if selected_indices is None:
            selected_indices = []
        if focused_indices is None:
            focused_indices = []

        if waveforms is None or len(waveforms) == 0:
            self.selected_plot.plot_waveforms(None, times, self.red_pen)
            self.focused_plot.plot_waveforms(None, times, self.blue_pen)
            self.unselected_plot.plot_waveforms(None, times, self.black_pen)
            return

        selection_mask = np.zeros(len(waveforms), dtype=bool)
        valid_indices = [i for i in selected_indices if i < len(waveforms)]
        if valid_indices:
            selection_mask[valid_indices] = True

        selected_waveforms = waveforms[selection_mask]
        unselected_waveforms = waveforms[~selection_mask]

        if selected_waveforms.shape[0] == 0:
            selected_waveforms = None
        if unselected_waveforms.shape[0] == 0:
            unselected_waveforms = None

        mean_selected = np.mean(selected_waveforms, axis=0) if selected_waveforms is not None else None
        self.selected_plot.plot_waveforms(selected_waveforms, times, self.red_pen,
                                          mean_waveform=mean_selected, mean_pen=self.green_pen)
        self.unselected_plot.plot_waveforms(unselected_waveforms, times, self.black_pen)

        if focused_indices:
            focus_mask = np.zeros(len(waveforms), dtype=bool)
            valid_focus_indices = [i for i in focused_indices if i < len(waveforms)]
            if valid_focus_indices:
                focus_mask[valid_focus_indices] = True

            focused_waveforms = waveforms[focus_mask]
            if focused_waveforms.shape[0] > 0:
                mean_focused = np.mean(focused_waveforms, axis=0)
                self.focused_plot.plot_waveforms(focused_waveforms, times, self.blue_pen,
                                                 mean_waveform=mean_focused, mean_pen=self.green_pen)
            else:
                self.focused_plot.plot_waveforms(None, times, self.blue_pen)
        else:
            self.focused_plot.plot_waveforms(None, times, self.blue_pen)


class ScaleFactorHistogramWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('w')
        plot_item = self.getPlotItem()
        plot_item.setLabel('bottom', 'Scale Factor (Std. Dev.)')
        plot_item.setLabel('left', 'Count')
        plot_item.showGrid(x=True, y=True)
        plot_item.getAxis('left').show()

        self.scale_factors = None
        self.unselected_fill = pg.mkBrush(100, 100, 100, 150)
        self.selected_fill = pg.mkBrush(255, 0, 0, 150)
        self.focused_fill = pg.mkBrush(0, 0, 255, 200)  # Blue with higher opacity

    def set_data(self, scale_factors):
        self.scale_factors = np.array(scale_factors)
        self.update_selection([], [])

    def update_selection(self, selected_indices, focused_indices=None):
        self.clear()
        if self.scale_factors is None or len(self.scale_factors) == 0:
            return

        if focused_indices is None:
            focused_indices = []

        selection_mask = np.zeros(len(self.scale_factors), dtype=bool)
        focused_mask = np.zeros(len(self.scale_factors), dtype=bool)

        valid_selected = [i for i in selected_indices if i < len(self.scale_factors)]
        valid_focused = [i for i in focused_indices if i < len(self.scale_factors)]

        if valid_selected:
            selection_mask[valid_selected] = True
        if valid_focused:
            focused_mask[valid_focused] = True

        # Create three categories with priority: focused > selected > unselected
        focused_scale_factors = self.scale_factors[focused_mask]
        selected_not_focused_mask = selection_mask & ~focused_mask
        selected_not_focused_scale_factors = self.scale_factors[selected_not_focused_mask]
        unselected_mask = ~selection_mask & ~focused_mask
        unselected_scale_factors = self.scale_factors[unselected_mask]

        if len(self.scale_factors) > 1:
            _, bin_edges = np.histogram(self.scale_factors, bins='sqrt')
        else:
            bin_edges = np.array([self.scale_factors[0]-0.5, self.scale_factors[0]+0.5]) if len(self.scale_factors) == 1 else np.array([0, 1])

        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[:-1] + bin_width / 2

        # Draw in order: unselected (bottom), selected (middle), focused (top)
        if len(unselected_scale_factors) > 0:
            y_unselected, _ = np.histogram(unselected_scale_factors, bins=bin_edges)
            unselected_bars = pg.BarGraphItem(
                x=bin_centers,
                height=y_unselected,
                width=bin_width,
                brush=self.unselected_fill,
                pen=pg.mkPen(None),
                zValue=1
            )
            self.addItem(unselected_bars)

        if len(selected_not_focused_scale_factors) > 0:
            y_selected, _ = np.histogram(selected_not_focused_scale_factors, bins=bin_edges)
            selected_bars = pg.BarGraphItem(
                x=bin_centers,
                height=y_selected,
                width=bin_width,
                brush=self.selected_fill,
                pen=pg.mkPen(None),
                zValue=2
            )
            self.addItem(selected_bars)

        if len(focused_scale_factors) > 0:
            y_focused, _ = np.histogram(focused_scale_factors, bins=bin_edges)
            focused_bars = pg.BarGraphItem(
                x=bin_centers,
                height=y_focused,
                width=bin_width,
                brush=self.focused_fill,
                pen=pg.mkPen(None),
                zValue=3
            )
            self.addItem(focused_bars)

class TimelineWidget(pg.PlotWidget):
    """
    Long, flat timeline showing trial alignment events and all spike times.
    - Trial align: thick green dashed vertical lines
    - Spikes: thin vertical lines (gray=unselected, red=selected, blue=focused)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('w')
        plot_item = self.getPlotItem()
        plot_item.setTitle("Spike and Event Timeline")
        plot_item.getAxis('left').setVisible(False)
        plot_item.showGrid(x=True, y=False)
        self.setYRange(0, 1.2)
        self.setMouseEnabled(x=True, y=False)

    def update_plot(self, spike_times, trial_align_times, selected_indices, focused_index=None):
        """
        Efficiently plots spikes and trial alignment events on a timeline.
        Spikes are drawn in batches to optimize performance for large datasets.
        Trial alignment markers are drawn at a separate height to ensure visibility.
        """
        self.clear()

        # 1. Prepare for batch-drawing spikes
        selected_set = set(selected_indices)
        focused_set = set()
        if focused_index is not None:
            if isinstance(focused_index, (list, set, np.ndarray)):
                focused_set = set(focused_index)
            else:
                focused_set.add(focused_index)

        unselected_times, selected_times, focused_times = [], [], []
        for i, t in enumerate(spike_times):
            if i in focused_set:
                focused_times.append(t)
            elif i in selected_set:
                selected_times.append(t)
            else:
                unselected_times.append(t)

        # 2. Helper function for batch drawing (now with y-range parameter)
        def batch_draw_lines(times, color, y_range=(0, 1), pen_style=QtCore.Qt.SolidLine, width=1):
            # Robust for lists and numpy arrays
            if len(times) == 0:
                return

            n_lines = len(times)
            x_coords = np.repeat(times, 2)
            y_coords = np.tile(y_range, n_lines)

            connect = np.ones(n_lines * 2, dtype=bool)
            connect[1::2] = 0

            pen = pg.mkPen(color, width=width, style=pen_style)
            item = pg.PlotDataItem(x=x_coords, y=y_coords, pen=pen, connect=connect)
            self.addItem(item)

        # 3. Draw trial alignment events FIRST (so they appear behind spikes if overlap)
        # They are drawn in the y=[1.0, 1.2] range.
        dark_green = (0, 150, 0)
        batch_draw_lines(trial_align_times, dark_green, y_range=(1.0, 1.2), pen_style=QtCore.Qt.DashLine, width=3)

        # 4. Draw each batch of spikes in the y=[0, 1] range
        batch_draw_lines(unselected_times, '#808080') # Gray
        batch_draw_lines(selected_times, 'r')         # Red
        batch_draw_lines(focused_times, 'b')         # Blue

        # 5. Set the view range
        all_times = np.concatenate([spike_times, trial_align_times])
        if all_times.size > 0:
            min_time, max_time = np.min(all_times), np.max(all_times)
            padding = (max_time - min_time) * 0.05
            self.setXRange(min_time - padding, max_time + padding, padding=0)



class MainWindow(QMainWindow):
    """
    Main window for the spike waveform curation tool.

    Provides an interactive interface for manually curating spike waveforms using
    UMAP visualization and elliptical selection tools. Supports both raw and
    z-scored waveform views with synchronized selection across both modes.
    """

    def __init__(self):
        """Initialize the main curation window."""
        super().__init__()
        self.setWindowTitle("Spike Waveform Curation Tool")
        self.setGeometry(100, 100, 1800, 900)

        self.node = None
        self.pkl_path = None
        self.umap_source = 'raw'
        self.umap_data_raw = None
        self.umap_data_zscored = None
        self.waveforms = None
        self.raw_waveforms = None
        self.times = None
        self.raw_times = None
        self.labels = None
        self.scale_factor = None
        self.trial_align = None
        self.spike_time = None

        # Store complete datasets for preserving unselected spikes
        self.complete_raw_waveforms = None
        self.complete_zscored_waveforms = None
        self.complete_scale_factor = None
        self.complete_raw_times = None
        self.complete_zscored_times = None

        # Mapping between filtered UMAP indices and original spike indices
        self.umap_to_original_indices = None

        self.ellipses_by_mode = {'raw': [], 'zscored': []}
        self.manual_selection_by_mode = {'raw': set(), 'zscored': set()}
        self.selected_indices_by_mode = {'raw': set(), 'zscored': set()}
        self.combined_selected_indices = []

        # Track whether each mode has been initialized with a default ellipse
        self.mode_initialized_with_default_ellipse = {'raw': False, 'zscored': False}

        self.pause_loop = None
        self.init_ui()

    def set_data_for_node(self, node: Node, pkl_path: str):
        self.node = node
        self.pkl_path = pkl_path
        self.setWindowTitle(f"Spike Curation Tool - Processing: {node.coordinate}")

        self.reset_ui_state()
        self.load_data()
        self.umap_plot.update_selection()
        # Initial draw of the timeline
        self.update_timeline_plot()

    def exec_blocking_loop(self):
        self.pause_loop = QEventLoop()
        self.pause_loop.exec_()

    def on_continue_clicked(self):
        """
        Connected to the 'Finish' button. Saves selection and quits the loop ONLY if saving is successful.
        """
        if self.save_selection():
            # Hide the window for a better user experience and quit the blocking loop.
            self.hide()
            if self.pause_loop:
                self.pause_loop.quit()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Exit Confirmation',
                                     "Are you sure you want to exit the curation process?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            logger.info("User aborted curation process.")
            if self.pause_loop:
                self.pause_loop.quit()
            QCoreApplication.instance().quit()
            sys.exit("Curation process terminated by user.")
        else:
            event.ignore()

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QGridLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.umap_plot = UMAPPlotWidget(self)
        self.umap_plot.getPlotItem().vb.setAspectLocked(True, ratio=1.0)
        self.zscored_waveform_plot = SplitWaveformWidget("Z-scored")
        self.raw_waveform_plot = SplitWaveformWidget("Raw")
        self.scale_factor_plot = ScaleFactorHistogramWidget()

        # Timeline visualization at the bottom
        self.timeline_widget = TimelineWidget()
        try:
            self.timeline_widget.setMaximumHeight(140)
        except Exception:
            pass

        button_layout = QHBoxLayout()

        self.add_ellipse_button = QPushButton("Add Ellipse")
        self.switch_umap_button = QPushButton("Switch to Z-Scored UMAP")
        self.update_umap_button = QPushButton("Update UMAP")
        self.continue_button = QPushButton("Save & Continue")
        self.load_button = QPushButton("Load Selection")

        button_layout.addWidget(self.add_ellipse_button)
        button_layout.addWidget(self.switch_umap_button)
        button_layout.addWidget(self.update_umap_button)
        button_layout.addWidget(self.continue_button)
        button_layout.addWidget(self.load_button)

        self.add_ellipse_button.clicked.connect(self.umap_plot.add_new_ellipse)
        self.switch_umap_button.clicked.connect(self.switch_umap_source)
        self.update_umap_button.clicked.connect(self.update_umap)
        self.continue_button.clicked.connect(self.on_continue_clicked)
        self.load_button.clicked.connect(self.load_selection)

        main_layout.addWidget(self.umap_plot, 0, 0)
        main_layout.addWidget(self.scale_factor_plot, 1, 0)
        main_layout.addWidget(self.zscored_waveform_plot, 0, 1, 2, 1)
        main_layout.addWidget(self.raw_waveform_plot, 0, 2, 2, 1)
        main_layout.addLayout(button_layout, 2, 0, 1, 3)

        # Timeline panel occupies full width at the bottom
        main_layout.addWidget(self.timeline_widget, 3, 0, 1, 3)

        main_layout.setColumnStretch(0, 1)
        main_layout.setColumnStretch(1, 1)
        main_layout.setColumnStretch(2, 1)
        main_layout.setRowStretch(0, 1)
        main_layout.setRowStretch(1, 1)

        self.statusBar().showMessage("Ready")

    def reset_ui_state(self):
        """Resets the UI controls and state variables to their default values."""
        self.umap_source = 'raw'
        self.switch_umap_button.setText("Switch to Z-Scored UMAP")

        self.ellipses_by_mode = {'raw': [], 'zscored': []}
        self.manual_selection_by_mode = {'raw': set(), 'zscored': set()}
        self.selected_indices_by_mode = {'raw': set(), 'zscored': set()}
        self.combined_selected_indices.clear()

        # Reset complete dataset storage
        self.complete_raw_waveforms = None
        self.complete_zscored_waveforms = None
        self.complete_scale_factor = None
        self.complete_raw_times = None
        self.complete_zscored_times = None
        self.umap_to_original_indices = None

        # Reset default ellipse tracking
        self.mode_initialized_with_default_ellipse = {'raw': False, 'zscored': False}

        self.statusBar().showMessage("Ready for new node.")

    def _calculate_umap(self, waveforms_data, description=""):
        """
        A centralized helper method to run UMAP on a given set of waveforms.
        Returns the 2D UMAP embedding.
        """
        if waveforms_data is None or len(waveforms_data) < 2:
            self.statusBar().showMessage(f"Not enough data to create UMAP for {description}.")
            return None

        self.statusBar().showMessage(f"Running UMAP on {len(waveforms_data)} spikes for {description}...")
        QApplication.processEvents()  # Allow the GUI to update the status message

        n_neighbors = min(15, int(np.sqrt(len(waveforms_data) - 1)))
        umap_model = UMAP(n_neighbors=n_neighbors, min_dist=0., n_components=2, metric='chebyshev')
        umap_embedding = umap_model.fit_transform(waveforms_data)

        self.statusBar().showMessage(f"UMAP for {description} calculated successfully.")
        return umap_embedding

    def load_data(self):
        self.statusBar().showMessage(f"Loading data for {self.node.coordinate}...")

        self.trial_align = self.node.timeline.advanced_filter(TRIAL_ALIGN_EVENT_DEFAULT).t
        self.spike_time = self.node.potential.spikes.t

        potential_timeseries = self.node.potential.aspect()
        spike_timeseries = potential_timeseries.batch_segment(self.node.potential.spikes.t,
                                                              CURATION_SPIKE_RANGE_RELATIVE_TO_ALIGNMENT)
        grouped_spike_timeseries = grouping_timeseries(spike_timeseries, interp_method="linear")
        self.raw_waveforms = grouped_spike_timeseries.raw_array
        self.raw_times = grouped_spike_timeseries.t

        zoom_in_indices = np.searchsorted(self.raw_times, SPIKE_RANGE_RELATIVE_TO_ALIGNMENT)
        zoom_in_waveforms = self.raw_waveforms[:, zoom_in_indices[0]:zoom_in_indices[1]]

        self.scale_factor = np.std(zoom_in_waveforms, axis=1) / (np.max(zoom_in_waveforms, axis=1) - np.min(zoom_in_waveforms, axis=1) + 1e-8)
        self.waveforms = zscore(zoom_in_waveforms, axis=1)
        self.times = self.raw_times[zoom_in_indices[0]:zoom_in_indices[1]]

        # Store complete datasets for preserving unselected spikes
        self.complete_raw_waveforms = self.raw_waveforms.copy()
        self.complete_zscored_waveforms = self.waveforms.copy()
        self.complete_scale_factor = self.scale_factor.copy()
        self.complete_raw_times = self.raw_times.copy()
        self.complete_zscored_times = self.times.copy()

        self.umap_data_raw = self._calculate_umap(zoom_in_waveforms, "raw waveforms")
        self.umap_data_zscored = self._calculate_umap(self.waveforms, "z-scored waveforms")

        # Initialize mapping - initially all spikes are included
        self.umap_to_original_indices = np.arange(len(self.waveforms))

        self.labels = KMeans(n_clusters=2).fit(self.umap_data_raw).labels_

        all_indices = set(range(len(self.waveforms)))
        self.selected_indices_by_mode = {
            'raw': all_indices.copy(),
            'zscored': all_indices.copy()
        }

        current_umap_data = self.umap_data_raw if self.umap_source == 'raw' else self.umap_data_zscored
        ellipses = self.ellipses_by_mode[self.umap_source]
        manual_selection = self.manual_selection_by_mode[self.umap_source]

        # Create default ellipse only for initial load in raw mode
        create_default = not self.mode_initialized_with_default_ellipse[self.umap_source]
        self.umap_plot.set_data(current_umap_data, self.labels, ellipses, manual_selection, create_default)

        # Mark this mode as initialized with default ellipse
        if create_default:
            self.mode_initialized_with_default_ellipse[self.umap_source] = True

        self.zscored_waveform_plot.plot_waveforms(self.complete_zscored_waveforms, self.complete_zscored_times)
        self.raw_waveform_plot.plot_waveforms(self.complete_raw_waveforms, self.complete_raw_times)
        self.scale_factor_plot.set_data(self.complete_scale_factor)

        self.statusBar().showMessage(f"Loaded {len(self.waveforms)} snippets. Mode: Opt-Out. Displaying UMAP on Z-Scored Waveforms.")

    def switch_umap_source(self):
        """Switch between raw and z-scored UMAP visualizations."""
        self.ellipses_by_mode[self.umap_source] = self.umap_plot.ellipses
        self.manual_selection_by_mode[self.umap_source] = self.umap_plot.manual_selection

        if self.umap_source == 'raw':
            self.umap_source = 'zscored'
            new_umap_data = self.umap_data_zscored
            self.switch_umap_button.setText("Switch to Raw UMAP")
            self.statusBar().showMessage("Switched to UMAP on Z-Scored Waveforms.")
        else:
            self.umap_source = 'raw'
            new_umap_data = self.umap_data_raw
            self.switch_umap_button.setText("Switch to Z-Scored UMAP")
            self.statusBar().showMessage("Switched to UMAP on Raw Waveforms.")

        new_ellipses = self.ellipses_by_mode[self.umap_source]
        new_manual_selection = self.manual_selection_by_mode[self.umap_source]

        # Create default ellipse only for first entry to this mode
        create_default = not self.mode_initialized_with_default_ellipse[self.umap_source]
        self.umap_plot.set_data(new_umap_data, self.labels, new_ellipses, new_manual_selection, create_default)

        # Mark this mode as initialized with default ellipse
        if create_default:
            self.mode_initialized_with_default_ellipse[self.umap_source] = True

    def on_selection_changed(self, current_view_indices):
        # Map UMAP view indices back to original spike indices if we're in filtered mode
        if self.umap_to_original_indices is not None and len(self.umap_to_original_indices) < len(self.complete_raw_waveforms):
            # We're in filtered UMAP mode - map back to original indices
            original_indices = [self.umap_to_original_indices[i] for i in current_view_indices if i < len(self.umap_to_original_indices)]

            # Update the selection for the current mode with original indices
            # But we need to be careful - we're only updating within the filtered subset
            self.selected_indices_by_mode[self.umap_source] = set(original_indices)
        else:
            # Normal mode - direct mapping
            self.selected_indices_by_mode[self.umap_source] = set(current_view_indices)

        raw_selection = self.selected_indices_by_mode['raw']
        zscored_selection = self.selected_indices_by_mode['zscored']
        self.combined_selected_indices = list(raw_selection.intersection(zscored_selection))

        # For color updates, we need to map back to the current UMAP view
        if self.umap_to_original_indices is not None and len(self.umap_to_original_indices) < len(self.complete_raw_waveforms):
            # Map combined selection back to UMAP view indices for coloring
            umap_view_indices = []
            for i, orig_idx in enumerate(self.umap_to_original_indices):
                if orig_idx in self.combined_selected_indices:
                    umap_view_indices.append(i)
            self.umap_plot.update_colors(umap_view_indices)
        else:
            self.umap_plot.update_colors(self.combined_selected_indices)

        # Calculate focused indices for the active ellipse
        focused_indices = []
        active_idx = self.umap_plot.active_ellipse_index
        if active_idx is not None and active_idx < len(self.umap_plot.ellipses):

            # Get all points inside the currently active ellipse (in UMAP view coordinates)
            points_in_active_ellipse = set(self.umap_plot.ellipses[active_idx].selected_indices)

            # Map to original indices if we're in filtered mode
            if self.umap_to_original_indices is not None and len(self.umap_to_original_indices) < len(self.complete_raw_waveforms):
                # Convert UMAP view indices to original indices
                original_points_in_active = set()
                for umap_idx in points_in_active_ellipse:
                    if umap_idx < len(self.umap_to_original_indices):
                        original_points_in_active.add(self.umap_to_original_indices[umap_idx])
                points_in_active_ellipse = original_points_in_active

            # Find all points opted-out by OTHER ellipses across BOTH modes
            other_opted_out_indices = set()

            # Check raw mode ellipses (excluding current active ellipse)
            for i, ellipse in enumerate(self.ellipses_by_mode['raw']):
                if self.umap_source != 'raw' or i != active_idx:
                    ellipse_indices = set(ellipse.selected_indices)
                    # Map to original indices if needed
                    if self.umap_to_original_indices is not None and len(self.umap_to_original_indices) < len(self.complete_raw_waveforms):
                        original_ellipse_indices = set()
                        for umap_idx in ellipse_indices:
                            if umap_idx < len(self.umap_to_original_indices):
                                original_ellipse_indices.add(self.umap_to_original_indices[umap_idx])
                        ellipse_indices = original_ellipse_indices
                    other_opted_out_indices.update(ellipse_indices)

            # Check z-scored mode ellipses (excluding current active ellipse)
            for i, ellipse in enumerate(self.ellipses_by_mode['zscored']):
                if self.umap_source != 'zscored' or i != active_idx:
                    ellipse_indices = set(ellipse.selected_indices)
                    # Map to original indices if needed
                    if self.umap_to_original_indices is not None and len(self.umap_to_original_indices) < len(self.complete_raw_waveforms):
                        original_ellipse_indices = set()
                        for umap_idx in ellipse_indices:
                            if umap_idx < len(self.umap_to_original_indices):
                                original_ellipse_indices.add(self.umap_to_original_indices[umap_idx])
                        ellipse_indices = original_ellipse_indices
                    other_opted_out_indices.update(ellipse_indices)

            # Focused waveforms are in active ellipse but not opted-out elsewhere
            focused_indices = list(points_in_active_ellipse.difference(other_opted_out_indices))

        # Always use complete datasets for waveform and histogram displays
        self.zscored_waveform_plot.plot_waveforms(self.complete_zscored_waveforms, self.complete_zscored_times, self.combined_selected_indices, focused_indices)
        self.raw_waveform_plot.plot_waveforms(self.complete_raw_waveforms, self.complete_raw_times, self.combined_selected_indices, focused_indices)
        self.scale_factor_plot.update_selection(self.combined_selected_indices, focused_indices)

        # Update the timeline visualization
        # FIX: Pass the entire list of focused indices to the timeline
        self.update_timeline_plot(focused_indices)

        self.statusBar().showMessage(f"Selected {len(self.combined_selected_indices)} snippets (intersection of both views)")

    def update_plots(self):
        """Refresh all visualizations based on the current manual selection mask only.
        This ignores any ellipse geometry and uses the intersection of manual selections
        across raw and z-scored modes as the final selection.
        """
        # Sync selected sets from manual selections
        for mode in ['raw', 'zscored']:
            sel = self.manual_selection_by_mode.get(mode, set())
            if not isinstance(sel, set):
                sel = set(sel)
            self.selected_indices_by_mode[mode] = sel

        raw_selection = self.selected_indices_by_mode['raw']
        zscored_selection = self.selected_indices_by_mode['zscored']
        self.combined_selected_indices = list(raw_selection.intersection(zscored_selection))

        # Update UMAP coloring in the current view
        if hasattr(self, 'umap_to_original_indices') and self.umap_to_original_indices is not None and \
           self.complete_raw_waveforms is not None and len(self.umap_to_original_indices) < len(self.complete_raw_waveforms):
            umap_view_indices = []
            for i, orig_idx in enumerate(self.umap_to_original_indices):
                if orig_idx in self.combined_selected_indices:
                    umap_view_indices.append(i)
            self.umap_plot.update_colors(umap_view_indices)
        else:
            self.umap_plot.update_colors(self.combined_selected_indices)

        # Calculate focused indices for the active ellipse (if any exist)
        focused_indices = []
        active_idx = self.umap_plot.active_ellipse_index
        if active_idx is not None and active_idx < len(self.umap_plot.ellipses):
            points_in_active_ellipse = set(self.umap_plot.ellipses[active_idx].selected_indices)
            # Map to original indices if we're in filtered mode
            if hasattr(self, 'umap_to_original_indices') and self.umap_to_original_indices is not None and \
               self.complete_raw_waveforms is not None and len(self.umap_to_original_indices) < len(self.complete_raw_waveforms):
                original_points_in_active = set()
                for umap_idx in points_in_active_ellipse:
                    if umap_idx < len(self.umap_to_original_indices):
                        original_points_in_active.add(self.umap_to_original_indices[umap_idx])
                points_in_active_ellipse = original_points_in_active

            # Points opted-out by other ellipses across both modes
            other_opted_out_indices = set()
            for i, ellipse in enumerate(self.ellipses_by_mode['raw']):
                if self.umap_source != 'raw' or i != active_idx:
                    ellipse_indices = set(ellipse.selected_indices)
                    if hasattr(self, 'umap_to_original_indices') and self.umap_to_original_indices is not None and \
                       self.complete_raw_waveforms is not None and len(self.umap_to_original_indices) < len(self.complete_raw_waveforms):
                        original_ellipse_indices = set()
                        for umap_idx in ellipse_indices:
                            if umap_idx < len(self.umap_to_original_indices):
                                original_ellipse_indices.add(self.umap_to_original_indices[umap_idx])
                        ellipse_indices = original_ellipse_indices
                    other_opted_out_indices.update(ellipse_indices)
            for i, ellipse in enumerate(self.ellipses_by_mode['zscored']):
                if self.umap_source != 'zscored' or i != active_idx:
                    ellipse_indices = set(ellipse.selected_indices)
                    if hasattr(self, 'umap_to_original_indices') and self.umap_to_original_indices is not None and \
                       self.complete_raw_waveforms is not None and len(self.umap_to_original_indices) < len(self.complete_raw_waveforms):
                        original_ellipse_indices = set()
                        for umap_idx in ellipse_indices:
                            if umap_idx < len(self.umap_to_original_indices):
                                original_ellipse_indices.add(self.umap_to_original_indices[umap_idx])
                        ellipse_indices = original_ellipse_indices
                    other_opted_out_indices.update(ellipse_indices)
            focused_indices = list(points_in_active_ellipse.difference(other_opted_out_indices))

        # Update waveform, histogram, and timeline using complete datasets
        if self.complete_zscored_waveforms is not None and self.complete_zscored_times is not None:
            self.zscored_waveform_plot.plot_waveforms(self.complete_zscored_waveforms, self.complete_zscored_times, self.combined_selected_indices, focused_indices)
        if self.complete_raw_waveforms is not None and self.complete_raw_times is not None:
            self.raw_waveform_plot.plot_waveforms(self.complete_raw_waveforms, self.complete_raw_times, self.combined_selected_indices, focused_indices)
        if hasattr(self, 'scale_factor_plot') and self.scale_factor_plot is not None:
            self.scale_factor_plot.update_selection(self.combined_selected_indices, focused_indices)

        # Timeline
        self.update_timeline_plot(focused_indices)

        self.statusBar().showMessage(f"Selected {len(self.combined_selected_indices)} snippets (intersection of both views)")

    def update_timeline_plot(self, focused_indices=None):
        """Helper to refresh the bottom timeline panel."""
        if not hasattr(self, 'timeline_widget') or self.timeline_widget is None:
            return
        if self.spike_time is None or self.trial_align is None:
            return
        selected_indices = self.combined_selected_indices if hasattr(self, 'combined_selected_indices') else []
        self.timeline_widget.update_plot(self.spike_time, self.trial_align, selected_indices, focused_index=focused_indices)


    def update_umap(self):
        if self.complete_raw_waveforms is None or self.complete_zscored_waveforms is None:
            self.statusBar().showMessage("No data loaded for UMAP update")
            return

        # Get current selection (intersection of both modes)
        raw_selection = self.selected_indices_by_mode['raw']
        zscored_selection = self.selected_indices_by_mode['zscored']
        selected_indices = list(raw_selection.intersection(zscored_selection))

        if len(selected_indices) < 2:
            self.statusBar().showMessage("Need at least 2 selected spikes for UMAP")
            return

        self.statusBar().showMessage(f"Updating UMAP with {len(selected_indices)} selected spikes...")

        # Extract selected spikes for UMAP computation
        zoom_in_indices = np.searchsorted(self.complete_raw_times, SPIKE_RANGE_RELATIVE_TO_ALIGNMENT)
        selected_raw_waveforms = self.complete_raw_waveforms[selected_indices]
        selected_raw_zoom = selected_raw_waveforms[:, zoom_in_indices[0]:zoom_in_indices[1]]
        selected_zscored_waveforms = self.complete_zscored_waveforms[selected_indices]

        # Compute UMAP on selected spikes only
        try:
            subset_umap_raw = self._calculate_umap(selected_raw_zoom, "selected subset (raw)")
            subset_umap_z = self._calculate_umap(selected_zscored_waveforms, "selected subset (z-scored)")
            if subset_umap_raw is None or subset_umap_z is None:
                return
            self.umap_data_raw = subset_umap_raw
            self.umap_data_zscored = subset_umap_z

            # Update mapping between UMAP indices and original spike indices
            self.umap_to_original_indices = np.array(selected_indices)

            # Update K-means clustering on the new UMAP data
            self.labels = KMeans(n_clusters=min(2, len(selected_indices))).fit(self.umap_data_raw).labels_

            # Clear ellipses since they're no longer valid for the new UMAP space
            self.ellipses_by_mode = {'raw': [], 'zscored': []}
            self.manual_selection_by_mode = {'raw': set(), 'zscored': set()}

            # Update the UMAP visualization - do NOT create default ellipses after Update UMAP
            current_umap_data = self.umap_data_raw if self.umap_source == 'raw' else self.umap_data_zscored
            self.umap_plot.set_data(current_umap_data, self.labels, [], set(), create_default_ellipse=False)

            # Reset selections to include all filtered spikes initially
            filtered_indices = set(range(len(selected_indices)))
            self.selected_indices_by_mode = {
                'raw': filtered_indices.copy(),
                'zscored': filtered_indices.copy()
            }

            self.statusBar().showMessage(f"UMAP updated with {len(selected_indices)} spikes. Ellipses cleared.")

        except Exception as e:
            self.statusBar().showMessage(f"Error updating UMAP: {str(e)}")

    def save_selection(self):
        """
        Saves the final, consolidated boolean selection mask to a pickle file,
        fulfilling the data contract with the downstream pipeline.
        Returns True on success, False on failure.
        """
        try:
            # --- START OF FIX ---
            # 1. Create the final boolean mask from the combined selected indices.
            # Prefer complete_raw_waveforms count; fallback to spike_time length.
            if self.complete_raw_waveforms is not None:
                num_spikes = int(self.complete_raw_waveforms.shape[0])
            elif self.spike_time is not None:
                num_spikes = int(len(self.spike_time))
            else:
                num_spikes = 0

            selected_mask = np.zeros(num_spikes, dtype=bool)

            if self.combined_selected_indices:
                # Ensure indices are valid before creating the mask.
                valid_indices = [idx for idx in self.combined_selected_indices if 0 <= idx < num_spikes]
                if len(valid_indices) > 0:
                    selected_mask[valid_indices] = True

            # 2. Create the data dictionary with the key the pipeline expects ('selected_mask').
            # We also save the internal state for potential reloading/debugging.
            data_to_save = {
                'selected_mask': selected_mask,
                'manual_selection_by_mode': self.manual_selection_by_mode,
            }
            # --- END OF FIX ---

            # 3. Ensure the output directory exists.
            dir_name = os.path.dirname(self.pkl_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)

            # 4. Save the correctly formatted data.
            with open(self.pkl_path, 'wb') as f:
                pickle.dump(data_to_save, f)

            self.statusBar().showMessage(f"Selection saved to {self.pkl_path}")
            return True

        except Exception as e:
            logging.error(f"Failed to save selection to {self.pkl_path}", exc_info=True)
            self.statusBar().showMessage(f"Error saving selection: {str(e)}")
            QMessageBox.critical(self, "Save Error", f"Could not save the selection file:\n\n{str(e)}")
            return False

    def load_selection(self):
        """Loads a manual selection mask and updates the plots."""
        if not os.path.exists(self.pkl_path):
            self.statusBar().showMessage("No selection file found to load.")
            return
        try:
            with open(self.pkl_path, 'rb') as f:
                data = pickle.load(f)
            # Load the core selection mask
            self.manual_selection_by_mode = data.get('manual_selection_by_mode', self.manual_selection_by_mode)
            # CRITICAL: Clear any existing ellipses from the plot's state
            for mode in self.ellipses_by_mode:
                self.ellipses_by_mode[mode] = []
            # Refresh all plots based on the loaded selection mask
            self.update_plots()
            self.statusBar().showMessage(f"Selection loaded from {self.pkl_path}")
        except Exception as e:
            self.statusBar().showMessage(f"Error loading selection: {str(e)}")


def node_spike_waveform_curation(node: Node, overwrite: bool):
    global _app_instance, _gui_instance

    pkl_path = routing.default_intermediate_result_path(node, result_name="spike_waveform_curation") + ".pkl"

    if (not os.path.exists(pkl_path)) or overwrite:
        if _app_instance is None:
            _app_instance = QApplication.instance() or QApplication(sys.argv)

        if _gui_instance is None:
            _gui_instance = MainWindow()

        _gui_instance.set_data_for_node(node, pkl_path)

        _gui_instance.show()
        _gui_instance.exec_blocking_loop()

    assert os.path.exists(pkl_path), f"Curation file not saved for {node.coordinate}. Process may have been aborted."
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    assert "selected_mask" in data
    selected_mask = data["selected_mask"]
    node.potential.spikes = node.potential.spikes.mask(selected_mask)
    logger.info(f"Mask applied. {np.sum(selected_mask)}/{len(selected_mask)} spikes selected for {node.coordinate}.")