import os.path
import sys
from typing import Optional
import numpy as np
import pickle
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QPushButton, QFileDialog, QMessageBox)
from PyQt5.QtCore import QPointF, QCoreApplication, QEventLoop
from PyQt5.QtGui import QTransform
import pyqtgraph as pg
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.cluster import KMeans
import logging

from kitchen.configs import routing
from kitchen.operator.grouping import grouping_timeseries
from kitchen.settings.potential import IMPORTANT_SPIKE_TIMEPOINTS, SPIKE_RANGE_RELATIVE_TO_ALIGNMENT
from kitchen.structure.hierarchical_data_structure import Node

pg.setConfigOptions(useOpenGL=True)

logger = logging.getLogger(__name__)

# --- Singleton instances for the GUI ---
_app_instance: Optional[QApplication] = None
_gui_instance: Optional['MainWindow'] = None


class SelectionEllipse:
    def __init__(self):
        self.center = QPointF(0, 0)
        self.width = 100
        self.height = 50
        self.rotation = 0.0
        self.selected_indices = []

    def contains_point(self, point):
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


class PCAPlotWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setBackground('w')
        self.points = None
        
        self.ellipses = []
        self.active_ellipse_index = None
        self.ellipse_items = []
        
        self.manual_selection = set()
        self.scatter = None
        self.resize_handle_item = None
        self.rotate_handle_item = None
        self.drag_start = None
        self.is_adjusting_ellipse = False
        self.initial_aspect_ratio = None

        self.getPlotItem().setLabel('bottom', 'UMAP 1')
        self.getPlotItem().setLabel('left', 'UMAP 2')
        self.getPlotItem().showGrid(x=True, y=True)

    def set_data(self, pc_data, labels):
        self.clear()
        self.points = pc_data
        self.labels = labels
        self.manual_selection.clear()

        self.brushes = [pg.mkBrush(100, 100, 100, 150) for _ in range(len(pc_data))]

        spots = [{'pos': pc_data[i], 'data': i} for i in range(len(pc_data))]
        self.scatter = pg.ScatterPlotItem(
            spots=spots,
            size=10,
            brush=self.brushes,
            pen=pg.mkPen(None),
            symbol='o'
        )
        self.addItem(self.scatter)

        self.ellipses.clear()
        initial_ellipse = SelectionEllipse()
        
        target_label = 0
        cluster_indices = np.where(self.labels == target_label)[0]

        if len(cluster_indices) > 0:
            points_for_ellipse = self.points[cluster_indices]
        else:
            points_for_ellipse = self.points

        if len(points_for_ellipse) > 0:
            center_x = np.mean(points_for_ellipse[:, 0])
            center_y = np.mean(points_for_ellipse[:, 1])
            initial_ellipse.center = QPointF(center_x, center_y)

            std_x = np.std(points_for_ellipse[:, 0])
            std_y = np.std(points_for_ellipse[:, 1])
            initial_ellipse.width = std_x * 5
            initial_ellipse.height = std_y * 5

        self.ellipses.append(initial_ellipse)
        self.active_ellipse_index = 0
        
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
        if self.resize_handle_item: self.removeItem(self.resize_handle_item)
        if self.rotate_handle_item: self.removeItem(self.rotate_handle_item)
        
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

        selection = set()
        if self.parent and self.parent.selection_mode == 'opt-out':
            all_indices = set(range(len(self.points)))
            selection = all_indices.difference(points_in_ellipses)
        else:
            selection = points_in_ellipses

        selection.update(self.manual_selection)

        brushes = self.brushes.copy()
        for i in range(len(self.points)):
            brushes[i] = pg.mkBrush(255, 0, 0, 200) if i in selection else self.brushes[i]

        self.scatter.setBrush(brushes)

        if self.parent:
            self.parent.on_selection_changed(list(selection))
            
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
    def __init__(self, parent=None, y_label="Amplitude"):
        super().__init__(parent)
        self.setBackground('w')
        plot_item = self.getPlotItem()
        plot_item.setLabel('bottom', 'Time (ms)')
        plot_item.setLabel('left', y_label)
        plot_item.showGrid(x=True, y=True)

    def plot_waveforms(self, waveforms, times, pen, mean_waveform=None, mean_pen=None):
        self.clear()
        if waveforms is None or len(waveforms) == 0:
            return

        max_waveforms_to_plot = 1000
        
        if len(waveforms) > max_waveforms_to_plot:
            indices_to_plot = np.random.choice(np.arange(len(waveforms)), max_waveforms_to_plot, replace=False)
            wfs_to_plot = waveforms[indices_to_plot]
        else:
            wfs_to_plot = waveforms

        x_vals = np.tile(np.append(times, np.nan), len(wfs_to_plot))
        y_vals = np.insert(wfs_to_plot, wfs_to_plot.shape[1], np.nan, axis=1).flatten()
        self.plot(x_vals, y_vals, pen=pen)

        if mean_waveform is not None and mean_pen is not None:
            self.plot(times, mean_waveform, pen=mean_pen)


class SplitWaveformWidget(QWidget):
    def __init__(self, data_type_name: str, parent=None):
        super().__init__(parent)

        self.selected_plot = WaveformPlotWidget(y_label=f"{data_type_name} Amplitude")
        # --- NEW: Add the focused plot widget ---
        self.focused_plot = WaveformPlotWidget(y_label=f"{data_type_name} Amplitude")
        self.unselected_plot = WaveformPlotWidget(y_label=f"{data_type_name} Amplitude")

        self.selected_plot.setTitle(f"Selected {data_type_name} Waveforms")
        # --- NEW: Set title for the focused plot ---
        self.focused_plot.setTitle(f"Focused Ellipse {data_type_name} Waveforms")
        self.unselected_plot.setTitle(f"Unselected {data_type_name} Waveforms")

        # Link Y-axes for consistent scaling
        self.focused_plot.setYLink(self.selected_plot)
        self.unselected_plot.setYLink(self.selected_plot)

        layout = QVBoxLayout()
        layout.addWidget(self.selected_plot)
        # --- NEW: Add the focused plot to the layout ---
        layout.addWidget(self.focused_plot)
        layout.addWidget(self.unselected_plot)
        self.setLayout(layout)

        self.black_pen = pg.mkPen(color=(0, 0, 0, 150), width=1)
        self.red_pen = pg.mkPen(color=(255, 0, 0, 150), width=1)
        # --- NEW: Add a blue pen for focused waveforms ---
        self.blue_pen = pg.mkPen(color=(0, 0, 255, 150), width=1)
        self.green_pen = pg.mkPen(color=(0, 255, 0), width=2)

    # --- MODIFIED: Update the plot_waveforms method signature and logic ---
    def plot_waveforms(self, waveforms, times, selected_indices=None, focused_indices=None):
        if selected_indices is None:
            selected_indices = []
        if focused_indices is None:
            focused_indices = []

        # Clear all plots if no waveforms are provided
        if waveforms is None or len(waveforms) == 0:
            self.selected_plot.plot_waveforms(None, times, self.red_pen)
            self.focused_plot.plot_waveforms(None, times, self.blue_pen)
            self.unselected_plot.plot_waveforms(None, times, self.black_pen)
            return

        # Plot selected and unselected waveforms (existing logic)
        selection_mask = np.zeros(len(waveforms), dtype=bool)
        valid_indices = [i for i in selected_indices if i < len(waveforms)]
        if valid_indices:
            selection_mask[valid_indices] = True

        selected_wfs = waveforms[selection_mask]
        unselected_wfs = waveforms[~selection_mask]

        if selected_wfs.shape[0] == 0:
            selected_wfs = None
        if unselected_wfs.shape[0] == 0:
            unselected_wfs = None

        mean_selected = np.mean(selected_wfs, axis=0) if selected_wfs is not None else None
        self.selected_plot.plot_waveforms(selected_wfs, times, self.red_pen,
                                          mean_waveform=mean_selected, mean_pen=self.green_pen)
        self.unselected_plot.plot_waveforms(unselected_wfs, times, self.black_pen)

        # --- NEW: Logic to plot waveforms from the focused ellipse ---
        if focused_indices:
            focus_mask = np.zeros(len(waveforms), dtype=bool)
            valid_focus_indices = [i for i in focused_indices if i < len(waveforms)]
            if valid_focus_indices:
                focus_mask[valid_focus_indices] = True

            focused_wfs = waveforms[focus_mask]
            if focused_wfs.shape[0] > 0:
                mean_focused = np.mean(focused_wfs, axis=0)
                self.focused_plot.plot_waveforms(focused_wfs, times, self.blue_pen,
                                                 mean_waveform=mean_focused, mean_pen=self.green_pen)
            else:
                self.focused_plot.plot_waveforms(None, times, self.blue_pen) # Clear if empty
        else:
            # If no ellipse is focused, clear the plot
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

    def set_data(self, scale_factors):
        self.scale_factors = np.array(scale_factors)
        self.update_selection([])

    def update_selection(self, selected_indices):
        self.clear()
        if self.scale_factors is None or len(self.scale_factors) == 0:
            return
            
        selection_mask = np.zeros(len(self.scale_factors), dtype=bool)
        valid_indices = [i for i in selected_indices if i < len(self.scale_factors)]
        if valid_indices:
            selection_mask[valid_indices] = True
        
        unselected_sf = self.scale_factors[~selection_mask]
        selected_sf = self.scale_factors[selection_mask]
        
        if len(self.scale_factors) > 1:
            _, bin_edges = np.histogram(self.scale_factors, bins='sqrt')
        else:
            bin_edges = np.array([self.scale_factors[0]-0.5, self.scale_factors[0]+0.5]) if len(self.scale_factors) == 1 else np.array([0, 1])

        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[:-1] + bin_width / 2

        if len(unselected_sf) > 0:
            y_unselected, _ = np.histogram(unselected_sf, bins=bin_edges)
            unselected_bars = pg.BarGraphItem(
                x=bin_centers, 
                height=y_unselected, 
                width=bin_width,
                brush=self.unselected_fill,
                pen=pg.mkPen(None)
            )
            self.addItem(unselected_bars)

        if len(selected_sf) > 0:
            y_selected, _ = np.histogram(selected_sf, bins=bin_edges)
            selected_bars = pg.BarGraphItem(
                x=bin_centers, 
                height=y_selected, 
                width=bin_width,
                brush=self.selected_fill,
                pen=pg.mkPen(None)
            )
            self.addItem(selected_bars)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spike Selection Tool")
        self.setGeometry(100, 100, 1800, 900)

        self.node = None
        self.pkl_path = None
        self.umap_source = 'raw' 
        self.selection_mode = 'opt-out' 
        self.pc_data_raw = None
        self.pc_data_zscored = None
        self.waveforms = None
        self.raw_waveforms = None
        self.times = None
        self.labels = None
        self.scale_factor = None
        self.selected_indices = []
        self.pause_loop = None

        self.init_ui()

    def set_data_for_node(self, node: Node, pkl_path: str):
        self.node = node
        self.pkl_path = pkl_path
        self.setWindowTitle(f"Spike Curation Tool - Processing: {node.coordinate}")
        
        self.reset_ui_state()
        
        self.load_data()
        self.pca_plot.update_selection()
        
    def exec_blocking_loop(self):
        self.pause_loop = QEventLoop()
        self.pause_loop.exec_()

    def on_continue_clicked(self):
        self.save_selection()
        if self.pause_loop:
            self.pause_loop.quit()
        self.hide()

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

        self.pca_plot = PCAPlotWidget(self)
        self.pca_plot.getPlotItem().vb.setAspectLocked(True, ratio=1.0)
        self.zscored_waveform_plot = SplitWaveformWidget("Z-scored")
        self.raw_waveform_plot = SplitWaveformWidget("Raw")
        self.scale_factor_plot = ScaleFactorHistogramWidget()

        button_layout = QHBoxLayout()

        self.add_ellipse_button = QPushButton("Add Ellipse")
        self.mode_switch_button = QPushButton("Switch to Opt-In")
        self.switch_umap_button = QPushButton("Switch to Z-Scored UMAP")
        self.update_button = QPushButton("Update Selection")
        self.continue_button = QPushButton("Save & Continue")
        self.load_button = QPushButton("Load Selection")

        button_layout.addWidget(self.add_ellipse_button)
        button_layout.addWidget(self.mode_switch_button)
        button_layout.addWidget(self.switch_umap_button)
        button_layout.addWidget(self.update_button)
        button_layout.addWidget(self.continue_button)
        button_layout.addWidget(self.load_button)

        self.add_ellipse_button.clicked.connect(self.pca_plot.add_new_ellipse)
        self.mode_switch_button.clicked.connect(self.switch_selection_mode)
        self.switch_umap_button.clicked.connect(self.switch_umap_source) 
        self.update_button.clicked.connect(self.pca_plot.update_selection)
        self.continue_button.clicked.connect(self.on_continue_clicked)
        self.load_button.clicked.connect(self.load_selection)

        main_layout.addWidget(self.pca_plot, 0, 0)
        main_layout.addWidget(self.scale_factor_plot, 1, 0)
        main_layout.addWidget(self.zscored_waveform_plot, 0, 1, 2, 1)
        main_layout.addWidget(self.raw_waveform_plot, 0, 2, 2, 1)
        main_layout.addLayout(button_layout, 2, 0, 1, 3)

        main_layout.setColumnStretch(0, 1)
        main_layout.setColumnStretch(1, 1)
        main_layout.setColumnStretch(2, 1)
        main_layout.setRowStretch(0, 1)
        main_layout.setRowStretch(1, 1)

        self.statusBar().showMessage("Ready")

    def reset_ui_state(self):
        """Resets the UI controls and state variables to their default values."""
        self.selection_mode = 'opt-out'
        self.mode_switch_button.setText("Switch to Opt-In")

        self.umap_source = 'raw'
        self.switch_umap_button.setText("Switch to Z-Scored UMAP")
        
        self.selected_indices.clear()
        
        self.statusBar().showMessage("Ready for new node.")

    def load_data(self):
        self.statusBar().showMessage(f"Loading data for {self.node.coordinate}...")
        
        potential_timeseries = self.node.potential.aspect()
        spike_timeseries = potential_timeseries.batch_segment(self.node.potential.spikes.t, 
                                                              SPIKE_RANGE_RELATIVE_TO_ALIGNMENT)
        grouped_spike_timeseries = grouping_timeseries(spike_timeseries, interp_method="linear")
        self.raw_waveforms = grouped_spike_timeseries.raw_array
        self.times = grouped_spike_timeseries.t

        self.scale_factor = np.std(self.raw_waveforms, axis=1)
        self.waveforms = (self.raw_waveforms - np.mean(self.raw_waveforms, axis=1, keepdims=True)) / (np.std(self.raw_waveforms, axis=1, keepdims=True) + 1e-8)

        umap_kwargs = {
            "n_components": 2,
            "n_neighbors": 15,
            "metric": 'chebyshev',
            "min_dist": 0.0,
            "n_epochs": 1000,
            # "metric_kwds": {"p": 4.0},
        }
        self.pc_data_raw = UMAP(**umap_kwargs).fit_transform(self.raw_waveforms)
        important_timepoint_index = np.searchsorted(self.times, IMPORTANT_SPIKE_TIMEPOINTS)
        self.pc_data_zscored = PCA(n_components=2).fit_transform(np.stack([self.waveforms[:, idx] for idx in important_timepoint_index], axis=1))
        
        self.labels = KMeans(n_clusters=2).fit(self.pc_data_raw).labels_

        current_pc_data = self.pc_data_raw if self.umap_source == 'raw' else self.pc_data_zscored
        self.pca_plot.set_data(current_pc_data, self.labels)
        
        self.zscored_waveform_plot.plot_waveforms(self.waveforms, self.times)
        self.raw_waveform_plot.plot_waveforms(self.raw_waveforms, self.times)
        self.scale_factor_plot.set_data(self.scale_factor)

        self.statusBar().showMessage(f"Loaded {len(self.waveforms)} snippets. Mode: Opt-Out. Displaying UMAP on Raw Waveforms.")

    def switch_selection_mode(self):
        if self.selection_mode == 'opt-out':
            self.selection_mode = 'opt-in'
            self.mode_switch_button.setText("Switch to Opt-Out")
            self.statusBar().showMessage("Switched to Opt-In mode. Points inside ellipses are selected.")
        else:
            self.selection_mode = 'opt-out'
            self.mode_switch_button.setText("Switch to Opt-In")
            self.statusBar().showMessage("Switched to Opt-Out mode. Points outside ellipses are selected.")
        
        self.pca_plot.update_selection()
        
    def switch_umap_source(self):
        if self.umap_source == 'raw':
            self.umap_source = 'zscored'
            self.pca_plot.set_data(self.pc_data_zscored, self.labels)
            self.switch_umap_button.setText("Switch to Raw UMAP")
            self.statusBar().showMessage("Switched to UMAP on Z-Scored Waveforms. Selection reset.")
        else:
            self.umap_source = 'raw'
            self.pca_plot.set_data(self.pc_data_raw, self.labels)
            self.switch_umap_button.setText("Switch to Z-Scored UMAP")
            self.statusBar().showMessage("Switched to UMAP on Raw Waveforms. Selection reset.")

    def on_selection_changed(self, selected_indices):
        self.selected_indices = selected_indices

        # --- NEW: Get indices for the currently focused ellipse ---
        focused_indices = []
        active_idx = self.pca_plot.active_ellipse_index
        if active_idx is not None and active_idx < len(self.pca_plot.ellipses):
            # The indices are already calculated and stored in the ellipse object
            focused_indices = self.pca_plot.ellipses[active_idx].selected_indices

        # --- MODIFIED: Pass focused_indices to the plotting methods ---
        self.zscored_waveform_plot.plot_waveforms(self.waveforms, self.times, selected_indices, focused_indices)
        self.raw_waveform_plot.plot_waveforms(self.raw_waveforms, self.times, selected_indices, focused_indices)

        self.scale_factor_plot.update_selection(selected_indices)
        self.statusBar().showMessage(f"Selected {len(selected_indices)} snippets")

    def save_selection(self):
        if not self.selected_indices:
            self.statusBar().showMessage("No selection to save")
            return

        selected_mask = np.zeros(len(self.waveforms), dtype=bool)
        selected_mask[self.selected_indices] = True
        
        ellipses_data = []
        for ellipse in self.pca_plot.ellipses:
            ellipses_data.append({
                'center_x': ellipse.center.x(),
                'center_y': ellipse.center.y(),
                'width': ellipse.width,
                'height': ellipse.height,
                'rotation': ellipse.rotation
            })

        data = {
            'selected_mask': selected_mask,
            'ellipses': ellipses_data,
            'manual_selection': list(self.pca_plot.manual_selection)
        }

        os.makedirs(os.path.dirname(self.pkl_path), exist_ok=True)
        with open(self.pkl_path, 'wb') as f:
            pickle.dump(data, f)

        self.statusBar().showMessage(f"Selection saved to {self.pkl_path}")

    def load_selection(self):
        try:
            with open(self.pkl_path, 'rb') as f:
                data = pickle.load(f)

            self.pca_plot.ellipses.clear()

            if 'ellipses' in data:
                for ellipse_data in data['ellipses']:
                    ellipse = SelectionEllipse()
                    ellipse.center = QPointF(ellipse_data['center_x'], ellipse_data['center_y'])
                    ellipse.width = ellipse_data['width']
                    ellipse.height = ellipse_data['height']
                    ellipse.rotation = ellipse_data['rotation']
                    self.pca_plot.ellipses.append(ellipse)
            
            elif 'ellipse' in data:
                ellipse_data = data['ellipse']
                ellipse = SelectionEllipse()
                ellipse.center = QPointF(ellipse_data['center_x'], ellipse_data['center_y'])
                ellipse.width = ellipse_data['width']
                ellipse.height = ellipse_data['height']
                ellipse.rotation = ellipse_data['rotation']
                self.pca_plot.ellipses.append(ellipse)

            if self.pca_plot.ellipses:
                self.pca_plot.active_ellipse_index = 0
            else:
                self.pca_plot.active_ellipse_index = None

            if 'manual_selection' in data:
                self.pca_plot.manual_selection = set(data['manual_selection'])

            self.pca_plot.draw_ellipses()
            self.pca_plot.update_selection()
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