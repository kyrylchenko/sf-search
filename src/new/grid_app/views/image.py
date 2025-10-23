import threading
import dearpygui.dearpygui as dpg
import array
import cv2
import numpy as np
from grid_app.other.pano_manipulation_service import PanoManipulationService
from grid_app.types.position_data import PositionData


class ImageView:
    def __init__(
        self,
        image_path: str,
    ) -> None:
        self.pano_manipulation_service = PanoManipulationService(image_path)
        self.image_ui_item: int | str | None = None
        self.highres_texture_height: int = 1080
        self.highres_texture_width: int = 1920
        self.lowres_texture_height: int = 225
        self.lowres_texture_width: int = 400
        self.fov = 80

        self.lowres_texture_ui_item: int | str | None = None
        self.highres_texture_ui_item: int | str | None = None

        self.highres_render_timer: threading.Timer | None = None

        self.highres_texture_data = np.frombuffer(
            self._get_perspective_1darray(
                0, 0, self.fov, self.highres_texture_height, self.highres_texture_width
            ),
            dtype=np.float32,
        )
        self.lowres_texture_data = np.frombuffer(
            self._get_perspective_1darray(
                0, 0, self.fov, self.lowres_texture_height, self.lowres_texture_width
            ),
            dtype=np.float32,
        )
        pass

    def _get_perspective_1darray(
        self, yaw: int, pitch: int, fov: int, image_height, image_width
    ):
        persp = self.pano_manipulation_service.get_perspective(
            fov, yaw, pitch, image_height, image_width
        )
        rgba = cv2.cvtColor(persp, cv2.COLOR_BGR2RGBA)  # BGR → RGBA
        rgba_float = rgba.astype(np.float32) / 255.0  # normalize if needed
        return array.array("f", np.ascontiguousarray(rgba_float).ravel())

    def setup(self):
        with dpg.texture_registry():
            self.lowres_texture_ui_item = dpg.add_raw_texture(
                width=self.lowres_texture_width,
                height=self.lowres_texture_height,
                default_value=self.lowres_texture_data,  # type: ignore
                format=dpg.mvFormat_Float_rgba,
                tag="lowres_texture",
            )
            self.highres_texture_ui_item = dpg.add_raw_texture(
                width=self.highres_texture_width,
                height=self.highres_texture_height,
                default_value=self.highres_texture_data,  # type: ignore
                format=dpg.mvFormat_Float_rgba,
                tag="highres_texture",
            )
        self.image_ui_item = dpg.add_image(
            texture_tag="highres_texture", width=1920, height=1080
        )
        pass

    def update_image(self, position_data: PositionData):
        if self.highres_render_timer is not None:
            self.highres_render_timer.cancel()

        self.lowres_texture_data[:] = self._get_perspective_1darray(
            position_data.yaw,
            position_data.pitch,
            position_data.fov,
            self.lowres_texture_height,
            self.lowres_texture_width,
        )

        assert self.image_ui_item is not None
        dpg.configure_item(self.image_ui_item, texture_tag="lowres_texture")
        self.highres_render_timer = threading.Timer(
            0.1,
            lambda: self._update_highres_image(
                position_data.yaw, position_data.pitch, position_data.fov
            ),
        )
        self.highres_render_timer.start()

    def _update_highres_image(self, x: int, y: int, fov: int):
        self.highres_texture_data[:] = self._get_perspective_1darray(
            x, y, fov, self.highres_texture_height, self.highres_texture_width
        )
        assert self.image_ui_item is not None
        dpg.configure_item(self.image_ui_item, texture_tag="highres_texture")
