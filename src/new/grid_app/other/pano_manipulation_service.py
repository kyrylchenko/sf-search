import cv2
import numpy as np


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
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_name}")
        self._img = img
        self._height, self._width, _ = self._img.shape

    def get_perspective(self, FOV, THETA, PHI, height, width) -> cv2.typing.MatLike:
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
        return cv2.remap(  # type ignore
            self._img,
            XY[..., 0],
            XY[..., 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
