import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QWidget,
    QSlider,
    QHBoxLayout,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage


# Optimize xyz2lonlat and lonlat2XY with precomputed constants and vectorized operations
def xyz2lonlat(xyz):
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    lon = np.arctan2(xyz_norm[..., 0], xyz_norm[..., 2])
    lat = np.arcsin(xyz_norm[..., 1])
    return np.stack((lon, lat), axis=-1)


def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1] / np.pi + 0.5) * (shape[0] - 1)
    return np.stack((X, Y), axis=-1).astype(np.float32)


class PanoManipulationService:
    def __init__(self, img_name: str):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        if self._img is None:
            raise FileNotFoundError(f"Image not found: {img_name}")
        self._height, self._width, _ = self._img.shape

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        f = 0.5 * width / np.tan(0.5 * np.radians(FOV))
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        K_inv = np.linalg.inv(
            np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
        )

        x, y = np.meshgrid(np.arange(width), np.arange(height))
        xyz = np.stack((x, y, np.ones_like(x)), axis=-1) @ K_inv.T

        R1, _ = cv2.Rodrigues(np.array([0, 1, 0], dtype=np.float32) * np.radians(THETA))
        R2, _ = cv2.Rodrigues(
            np.dot(R1, np.array([1, 0, 0], dtype=np.float32)) * np.radians(PHI)
        )
        xyz = xyz @ (R2 @ R1).T

        lonlat = xyz2lonlat(xyz)
        XY = lonlat2XY(lonlat, self._img.shape)
        return cv2.remap(
            self._img,
            XY[..., 0],
            XY[..., 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )


class PanoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pano Manipulation Viewer")
        self.setGeometry(100, 100, 1080, 720)

        # Parameters
        self.fov, self.theta, self.phi = 100, 0, 0
        self.width, self.height = 1080, 720

        # PanoManipulationService
        self.pano_service = PanoManipulationService(
            "/Users/illia/Documents/projects/pano-analyzer/panos-new/s3JP0Zf-5IGwBd-qmX4HmA.jpg"
        )

        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout()

        # Image display
        self.image_label = QLabel(alignment=Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # Sliders
        self.slider_layout = QHBoxLayout()
        self.fov_slider = self.create_slider(10, 170, self.fov, self.update_image)
        self.theta_slider = self.create_slider(-180, 180, self.theta, self.update_image)
        self.phi_slider = self.create_slider(-90, 90, self.phi, self.update_image)
        self.slider_layout.addWidget(self.fov_slider)
        self.slider_layout.addWidget(self.theta_slider)
        self.slider_layout.addWidget(self.phi_slider)
        self.layout.addLayout(self.slider_layout)

        self.main_widget.setLayout(self.layout)
        self.update_image()

    def create_slider(self, min_val, max_val, init_val, callback):
        slider = QSlider(
            Qt.Horizontal, minimum=min_val, maximum=max_val, value=init_val
        )
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(10)
        slider.valueChanged.connect(callback)
        return slider

    def update_image(self):
        # Ensure slider values are integers
        self.fov = int(self.fov_slider.value())
        self.theta = int(self.theta_slider.value())
        self.phi = int(self.phi_slider.value())

        # Generate new perspective image
        new_img = self.pano_service.GetPerspective(
            self.fov, self.theta, self.phi, self.height, self.width
        )

        # Convert to QImage and display
        q_img = QImage(
            new_img.data,
            new_img.shape[1],
            new_img.shape[0],
            new_img.strides[0],
            QImage.Format_RGB888,
        ).rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def wheelEvent(self, event):
        self.fov = max(10, min(170, self.fov - event.angleDelta().y() / 120 * 5))
        self.fov_slider.setValue(self.fov)
        self.update_image()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_position = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            delta = event.pos() - self.last_mouse_position
            self.theta = int((self.theta + delta.x() * 0.5) % 360)
            self.phi = int(max(-90, min(90, self.phi - delta.y() * 0.5)))
            self.last_mouse_position = event.pos()
            self.theta_slider.setValue(self.theta)
            self.phi_slider.setValue(self.phi)
            self.update_image()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PanoApp()
    window.show()
    sys.exit(app.exec_())
