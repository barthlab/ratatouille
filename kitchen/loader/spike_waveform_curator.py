import os.path
import sys
import numpy as np
import pickle
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QPushButton, QLabel, QFileDialog)
from PyQt5.QtCore import Qt, QPointF, QCoreApplication
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QTransform, QPainterPath
import pyqtgraph as pg
pg.setConfigOptions(useOpenGL=True)

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.spatial import distance

# from src.basic.utils import *
# from src.basic.config import *


class SelectionEllipse:
    def __init__(self):
        self.center = QPointF(0, 0)
        self.width = 100
        self.height = 50
        self.rotation = 0.0
        self.selected_indices = []

    def contains_point(self, point):
        # Transform point to ellipse coordinate system
        transform = QTransform()
        transform.translate(self.center.x(), self.center.y())
        transform.rotate(self.rotation)
        transformed_point = transform.inverted()[0].map(point)

        # Check if point is inside ellipse (no extra subtraction needed)
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

        for i, point in enumerate(points):
            transformed_point = inv.map(QPointF(point[0], point[1]))
            normalized_x = transformed_point.x() / (self.width / 2)
            normalized_y = transformed_point.y() / (self.height / 2)
            if normalized_x ** 2 + normalized_y ** 2 <= 1:
                self.selected_indices.append(i)
        return self.selected_indices

    def get_path(self):
        path = QPainterPath()
        transform = QTransform()
        transform.translate(self.center.x(), self.center.y())
        transform.rotate(self.rotation)
        path.addEllipse(QPointF(0, 0), self.width / 2, self.height / 2)
        return transform.map(path)


class PCAPlotWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setBackground('w')
        self.points = None
        self.ellipse = SelectionEllipse()
        self.manual_selection = set()
        self.scatter = None
        self.ellipse_item = None
        self.resize_handle_item = None
        self.rotate_handle_item = None
        self.drag_start = None
        self.is_adjusting_ellipse = False
        self.initial_aspect_ratio = None  # For fixed aspect ratio during resize

        # Setup plot
        self.getPlotItem().setLabel('bottom', 'PC1')
        self.getPlotItem().setLabel('left', 'PC2')
        self.getPlotItem().showGrid(x=True, y=True)

    def set_data(self, pc_data, labels):
        self.clear()
        self.points = pc_data
        self.labels = labels

        # Create scatter plot using spots with index data
        unique_labels = np.unique(labels)
        colors = [pg.mkColor(i * 30, 255, 255, 150) for i in range(len(unique_labels))]

        # Initialize brushes for different clusters
        self.brushes = []
        for i in range(len(pc_data)):
            if labels[i] == -1:  # Noise points
                self.brushes.append(pg.mkBrush(100, 100, 100, 150))
            else:
                self.brushes.append(pg.mkBrush(colors[labels[i] % len(colors)]))

        spots = [{'pos': pc_data[i], 'data': i} for i in range(len(pc_data))]
        self.scatter = pg.ScatterPlotItem(
            spots=spots,
            size=10,
            brush=self.brushes,
            pen=pg.mkPen(None),
            symbol='o'
        )
        # self.scatter.sigClicked.connect(self.on_point_clicked)
        self.addItem(self.scatter)

        # Center the ellipse on data
        center_x = np.mean(pc_data[:, 0])
        center_y = np.mean(pc_data[:, 1])
        self.ellipse.center = QPointF(center_x, center_y)

        # Set ellipse size based on data spread
        std_x = np.std(pc_data[:, 0]) * 2
        std_y = np.std(pc_data[:, 1]) * 2
        self.ellipse.width = std_x * 2
        self.ellipse.height = std_y * 2

        # Create ellipse
        self.draw_ellipse()

        # Add handles for resizing and rotating
        self.add_control_handles()

        # Update selection
        self.update_selection()

    def draw_ellipse(self):
        if self.ellipse_item is not None:
            self.removeItem(self.ellipse_item)

        # Create ellipse path manually
        path_points = []
        cos_rot = np.cos(np.radians(self.ellipse.rotation))
        sin_rot = np.sin(np.radians(self.ellipse.rotation))
        for i in range(100):
            angle = 2 * np.pi * i / 100
            # Start with a point on the unrotated ellipse
            x = self.ellipse.width / 2 * np.cos(angle)
            y = self.ellipse.height / 2 * np.sin(angle)

            # Rotate the point
            rotated_x = self.ellipse.center.x() + x * cos_rot - y * sin_rot
            rotated_y = self.ellipse.center.y() + x * sin_rot + y * cos_rot

            path_points.append((rotated_x, rotated_y))

        path_points = np.array(path_points)
        self.ellipse_item = pg.PlotCurveItem(
            pen=pg.mkPen(color=(0, 0, 255), width=2)
        )
        self.ellipse_item.setData(path_points[:, 0], path_points[:, 1])
        self.addItem(self.ellipse_item)

    def add_control_handles(self):
        # Clear existing handles
        if self.resize_handle_item is not None:
            self.removeItem(self.resize_handle_item)
        if self.rotate_handle_item is not None:
            self.removeItem(self.rotate_handle_item)

        # Add resize handle (right side of ellipse)
        right_point_x = self.ellipse.center.x() + self.ellipse.width / 2 * np.cos(np.radians(self.ellipse.rotation))
        right_point_y = self.ellipse.center.y() + self.ellipse.width / 2 * np.sin(np.radians(self.ellipse.rotation))
        self.resize_handle_pos = np.array([[right_point_x, right_point_y]])
        self.resize_handle_item = pg.ScatterPlotItem(
            pos=self.resize_handle_pos,
            size=15,
            brush=pg.mkBrush(255, 0, 0, 200),
            pen=pg.mkPen('w'),
            symbol='s'
        )
        self.resize_handle_item.setZValue(10)
        self.addItem(self.resize_handle_item)

        # Add rotation handle (top side of ellipse)
        top_point_x = self.ellipse.center.x() + self.ellipse.height / 2 * np.cos(np.radians(self.ellipse.rotation + 90))
        top_point_y = self.ellipse.center.y() + self.ellipse.height / 2 * np.sin(np.radians(self.ellipse.rotation + 90))
        self.rotate_handle_pos = np.array([[top_point_x, top_point_y]])
        self.rotate_handle_item = pg.ScatterPlotItem(
            pos=self.rotate_handle_pos,
            size=15,
            brush=pg.mkBrush(0, 255, 0, 200),
            pen=pg.mkPen('w'),
            symbol='o'
        )
        self.rotate_handle_item.setZValue(10)
        self.addItem(self.rotate_handle_item)

    def update_selection(self):

        if self.points is None:
            return

        # Update selection based on ellipse
        ellipse_selection = set(self.ellipse.update_selection(self.points))

        # Add manually selected points
        selection = ellipse_selection.union(self.manual_selection)

        # Update scatter plot colors
        brushes = self.brushes.copy()
        for i in range(len(self.points)):
            if i in selection:
                brushes[i] = pg.mkBrush(255, 0, 0, 200)  # Selected points in red

        self.scatter.setBrush(brushes)

        # Notify parent
        if self.parent:
            self.parent.on_selection_changed(list(selection))

    # def on_point_clicked(self, plot, points):
    #     for point in points:
    #         index = point.data()  # Retrieve the index stored in the spot's data
    #         # Toggle point selection
    #         if index in self.manual_selection:
    #             self.manual_selection.remove(index)
    #         else:
    #             self.manual_selection.add(index)

    def mousePressEvent(self, event):
        pos = self.plotItem.vb.mapSceneToView(event.pos())

        # Check if we're clicking on a control handle
        resize_point = QPointF(self.resize_handle_pos[0][0], self.resize_handle_pos[0][1])
        rotate_point = QPointF(self.rotate_handle_pos[0][0], self.rotate_handle_pos[0][1])

        # Distance thresholds
        threshold = 20  # pixels

        # Convert to screen coordinates for distance check
        screen_pos = self.plotItem.vb.mapViewToScene(pos)
        screen_resize = self.plotItem.vb.mapViewToScene(resize_point)
        screen_rotate = self.plotItem.vb.mapViewToScene(rotate_point)

        # Check distances
        resize_dist = (screen_pos - screen_resize).manhattanLength()
        rotate_dist = (screen_pos - screen_rotate).manhattanLength()

        if resize_dist < threshold:
            self.is_adjusting_ellipse = 'resize'
            self.resize_initial_width = self.ellipse.width
            self.resize_initial_height = self.ellipse.height
            transform = QTransform().rotate(-self.ellipse.rotation)
            self.initial_local = transform.map(pos - self.ellipse.center)
        elif rotate_dist < threshold:
            self.is_adjusting_ellipse = 'rotate'
            self.resize_initial_width = self.ellipse.width
            self.resize_initial_height = self.ellipse.height
            transform = QTransform().rotate(-self.ellipse.rotation)
            self.initial_local = transform.map(pos - self.ellipse.center)
        else:
            # Check if clicking the ellipse for dragging
            if self.ellipse.contains_point(pos):
                self.is_adjusting_ellipse = 'drag'
            else:
                self.is_adjusting_ellipse = False

        self.drag_start = pos
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag_start is None or not self.is_adjusting_ellipse:
            super().mouseMoveEvent(event)
            return

        pos = self.plotItem.vb.mapSceneToView(event.pos())
        dx = pos.x() - self.drag_start.x()
        dy = pos.y() - self.drag_start.y()

        if self.is_adjusting_ellipse == 'drag':
            # Move the ellipse center
            self.ellipse.center = QPointF(self.ellipse.center.x() + dx, self.ellipse.center.y() + dy)
            self.drag_start = pos

        elif self.is_adjusting_ellipse == 'resize':
            transform = QTransform().rotate(-self.ellipse.rotation)
            current_local = transform.map(pos - self.ellipse.center)
            if self.initial_local.x() != 0:
                scale_factor = abs(current_local.x() / self.initial_local.x())
            else:
                scale_factor = 1
            self.ellipse.width = self.resize_initial_width * scale_factor
            self.ellipse.height = self.resize_initial_height

            self.drag_start = pos

        elif self.is_adjusting_ellipse == 'rotate':
            transform = QTransform().rotate(-self.ellipse.rotation)
            current_local = transform.map(pos - self.ellipse.center)

            if self.initial_local.x() != 0:
                scale_factor = abs(current_local.y() / self.initial_local.y())
            else:
                scale_factor = 1
            self.ellipse.width = self.resize_initial_width
            self.ellipse.height = self.resize_initial_height * scale_factor

            # Calculate vectors for rotation angle
            dx = pos.x() - self.ellipse.center.x()
            dy = pos.y() - self.ellipse.center.y()
            self.ellipse.rotation = np.degrees(np.arctan2(dy, dx) - 90) % 360

            self.drag_start = pos

        # Redraw ellipse and update handles and selection
        self.draw_ellipse()
        self.add_control_handles()

        event.accept()

    def mouseReleaseEvent(self, event):
        self.drag_start = None
        self.is_adjusting_ellipse = False
        super().mouseReleaseEvent(event)


class WaveformPlotWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('w')
        plot_item = self.getPlotItem()
        plot_item.setLabel('bottom', 'Time')
        plot_item.setLabel('left', 'Amplitude')
        plot_item.showGrid(x=True, y=True)

    def plot_waveforms(self, waveforms, times, selected_indices=None):
        self.clear()

        # Convert selected_indices to a set for faster membership tests
        selected_set = set(selected_indices) if selected_indices is not None else set()

        # Pre-create pens to avoid recreating them in loops
        light_pen = pg.mkPen(color=(200, 200, 200, 100), width=2)
        red_pen = pg.mkPen(color=(255, 0, 0, 20))
        green_pen = pg.mkPen(color=(0, 255, 0), width=2)

        # Plot selected waveforms in red
        plotting_index = np.random.choice(len(selected_set), 500, replace=False)\
            if len(selected_set) > 500 else range(len(selected_set))
        for idx in plotting_index:
            self.plot(times, waveforms[list(selected_set)[idx]]+1, pen=red_pen)

        # Plot unselected waveforms in light gray
        unselected = []
        for i, waveform in enumerate(waveforms):
            if i in selected_set:
                continue
            unselected.append(i)
        plotting_index = np.random.choice(len(unselected), 500, replace=False) \
            if len(unselected) > 500 else range(len(unselected))
        for idx in plotting_index:
            self.plot(times, waveforms[list(unselected)[idx]], pen=light_pen)

        # Plot mean of selected waveforms in green, if any are selected
        if selected_set:
            selected_waveforms = [waveforms[i] for i in selected_set]
            mean_selected = np.mean(selected_waveforms, axis=0)
            self.plot(times, mean_selected+1, pen=green_pen)


class MainWindow(QMainWindow):
    def __init__(self, snippets, pkl_path):
        super().__init__()
        self.setWindowTitle("Spike Selection Tool")
        self.setGeometry(100, 100, 1200, 800)

        # State variables
        self.snippets = snippets
        self.pkl_path = pkl_path
        self.pc_data = None
        self.waveforms = None
        self.times = None
        self.labels = None
        self.selected_indices = []

        self.init_ui()
        self.load_data()

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QGridLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Create plot widgets
        self.pca_plot = PCAPlotWidget(self)
        self.pca_plot.getPlotItem().vb.setAspectLocked(True, ratio=1.0)
        self.waveform_plot = WaveformPlotWidget()

        # Create buttons
        button_layout = QHBoxLayout()
        self.update_button = QPushButton("Update Selection")
        self.save_button = QPushButton("Save Selection")
        self.load_button = QPushButton("Load Selection")
        # self.close_button = QPushButton("Close")

        button_layout.addWidget(self.update_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.load_button)
        # button_layout.addWidget(self.close_button)

        # Connect signals
        self.update_button.clicked.connect(self.pca_plot.update_selection)
        self.save_button.clicked.connect(self.save_selection)
        self.load_button.clicked.connect(self.load_selection)
        # self.close_button.clicked.connect(QCoreApplication.instance().quit)

        # Set up layout
        main_layout.addWidget(self.pca_plot, 0, 0)
        main_layout.addWidget(self.waveform_plot, 0, 1)
        main_layout.addLayout(button_layout, 1, 0, 1, 2)

        # Status bar
        self.statusBar().showMessage("Ready")

    def load_data(self):
        total_times = [single_snippet.data.vm_t_aligned
                       for single_snippet in self.snippets]
        total_data = [single_snippet.data.spectra[SPIKING_BANDWIDTH]
                      for single_snippet in self.snippets]
        xs, _, _, interp_value = synchronize_time_series_data(total_times, total_data)
        self.waveforms = np.array(interp_value)
        self.times = xs

        # Process data
        self.pc_data = PCA(n_components=2).fit_transform(self.waveforms)
        self.labels = DBSCAN().fit(self.pc_data).labels_

        # Update plots
        self.pca_plot.set_data(self.pc_data, self.labels)
        self.waveform_plot.plot_waveforms(self.waveforms, self.times)

        self.statusBar().showMessage(f"Loaded {len(self.waveforms)} snippets")

    def on_selection_changed(self, selected_indices):
        self.selected_indices = selected_indices
        self.waveform_plot.plot_waveforms(self.waveforms, self.times, selected_indices)
        self.statusBar().showMessage(f"Selected {len(selected_indices)} snippets")

    def save_selection(self):
        if self.selected_indices is None:
            self.statusBar().showMessage("No selection to save")
            return

        selected_mask = np.zeros(len(self.snippets), dtype=bool)
        for selected_id in self.selected_indices:
            selected_mask[selected_id] = True
        data = {
            'selected_mask': selected_mask,
            'ellipse': {
                'center_x': self.pca_plot.ellipse.center.x(),
                'center_y': self.pca_plot.ellipse.center.y(),
                'width': self.pca_plot.ellipse.width,
                'height': self.pca_plot.ellipse.height,
                'rotation': self.pca_plot.ellipse.rotation
            },
            'manual_selection': list(self.pca_plot.manual_selection)
        }

        with open(self.pkl_path, 'wb') as f:
            pickle.dump(data, f)

        self.statusBar().showMessage(f"Selection saved to {self.pkl_path}")

    def load_selection(self):
        try:
            with open(self.pkl_path, 'rb') as f:
                data = pickle.load(f)

            # Restore ellipse
            if 'ellipse' in data:
                ellipse_data = data['ellipse']
                self.pca_plot.ellipse.center = QPointF(
                    ellipse_data['center_x'],
                    ellipse_data['center_y']
                )
                self.pca_plot.ellipse.width = ellipse_data['width']
                self.pca_plot.ellipse.height = ellipse_data['height']
                self.pca_plot.ellipse.rotation = ellipse_data['rotation']

                self.pca_plot.draw_ellipse()
                self.pca_plot.add_control_handles()

            # Restore manual selection
            if 'manual_selection' in data:
                self.pca_plot.manual_selection = set(data['manual_selection'])

            self.pca_plot.update_selection()
            self.statusBar().showMessage(f"Selection loaded from {self.pkl_path}")
        except Exception as e:
            self.statusBar().showMessage(f"Error loading selection: {str(e)}")


app = QApplication(sys.argv)


def get_selection(snippets, pkl_path):
    if not os.path.exists(pkl_path):
        gui = MainWindow(snippets, pkl_path)
        gui.show()
        app.exec_()
    assert os.path.exists(pkl_path)

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    assert "selected_mask" in data
    return data["selected_mask"]