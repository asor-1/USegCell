import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QListWidget
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint
import numpy as np
from skimage import io
from PIL import Image

class CellSegmentationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cell Segmentation GUI")
        self.setGeometry(100, 100, 800, 600)

        self.tab_widget = QTabWidget(self)
        self.setCentralWidget(self.tab_widget)

        self.segmentation_tab = SegmentationTab()
        self.tab_widget.addTab(self.segmentation_tab, "Manual Segmentation")

        # TODO: Add training tab in the future
        # self.training_tab = TrainingTab()
        # self.tab_widget.addTab(self.training_tab, "Model Training")

class SegmentationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QHBoxLayout()

        # Left panel for image list and controls
        left_panel = QVBoxLayout()
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.load_image)
        left_panel.addWidget(QLabel("Images:"))
        left_panel.addWidget(self.image_list)

        self.load_images_button = QPushButton("Load Images")
        self.load_images_button.clicked.connect(self.load_image_directory)
        left_panel.addWidget(self.load_images_button)

        self.save_segmentation_button = QPushButton("Save Segmentation")
        self.save_segmentation_button.clicked.connect(self.save_segmentation)
        left_panel.addWidget(self.save_segmentation_button)

        self.clear_segmentation_button = QPushButton("Clear Segmentation")
        self.clear_segmentation_button.clicked.connect(self.clear_segmentation)
        left_panel.addWidget(self.clear_segmentation_button)

        # Right panel for image display and segmentation
        right_panel = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.image_label)

        layout.addLayout(left_panel, 1)
        layout.addLayout(right_panel, 3)
        self.setLayout(layout)

        self.current_image = None
        self.segmentation_mask = None
        self.drawing = False
        self.last_point = None

    def load_image_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if directory:
            self.image_list.clear()
            for filename in os.listdir(directory):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    self.image_list.addItem(os.path.join(directory, filename))

    def load_image(self, item):
        image_path = item.text()
        self.current_image = io.imread(image_path)
        self.display_image()
        self.segmentation_mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)

    def display_image(self):
        if self.current_image is not None:
            height, width = self.current_image.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(self.current_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)
            self.image_label.setFixedSize(pixmap.size())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.current_image is not None:
            self.drawing = True
            self.last_point = event.pos() - self.image_label.pos()
            self.draw_segmentation(event)

    def mouseMoveEvent(self, event):
        if self.drawing and self.current_image is not None:
            self.draw_segmentation(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.last_point = None

    def draw_segmentation(self, event):
        if self.current_image is None:
            return

        current_point = event.pos() - self.image_label.pos()
        painter = QPainter(self.image_label.pixmap())
        painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.SolidLine))
        
        if self.last_point:
            painter.drawLine(self.last_point, current_point)
        else:
            painter.drawPoint(current_point)

        self.last_point = current_point
        self.image_label.update()

        # Update segmentation mask
        mask_painter = QPainter(QPixmap.fromImage(QImage(self.segmentation_mask.data, self.segmentation_mask.shape[1], self.segmentation_mask.shape[0], QImage.Format_Grayscale8)))
        mask_painter.setPen(QPen(QColor(255, 255, 255), 2, Qt.SolidLine))
        if self.last_point:
            mask_painter.drawLine(self.last_point, current_point)
        else:
            mask_painter.drawPoint(current_point)
        self.segmentation_mask = np.array(mask_painter.device().toImage())

    def save_segmentation(self):
        if self.segmentation_mask is not None:
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Segmentation", "", "NumPy Files (*.npy)")
            if file_name:
                np.save(file_name, self.segmentation_mask)
                print(f"Segmentation saved to {file_name}")

    def clear_segmentation(self):
        if self.current_image is not None:
            self.segmentation_mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
            self.display_image()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = CellSegmentationGUI()
    gui.show()
    sys.exit(app.exec_())