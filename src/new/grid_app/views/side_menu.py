import dearpygui.dearpygui as dpg
from collections.abc import Callable
from grid_app.types.position_data import PositionData


class SideMenuView:
    def __init__(
        self, on_position_value_update: Callable[[PositionData], None]
    ) -> None:
        self.on_position_value_update = on_position_value_update

        self.pitch_slider_value: int = 0
        self.yaw_slider_value: int = 0
        self.fov_slider_value: int = 90

        self.pitch_slider_ui_item: int | str | None = None
        self.yaw_slider_ui_item: int | str | None = None
        self.fov_slider_ui_item: int | str | None = None
        pass

    def setup(self) -> None:
        self.pitch_slider_ui_item = dpg.add_slider_int(
            label="Pitch",
            min_value=-180,
            max_value=180,
            default_value=0,
            callback=self._on_position_value_update,
        )
        self.yaw_slider_ui_item = dpg.add_slider_int(
            label="Yaw",
            min_value=-180,
            max_value=180,
            default_value=0,
            callback=self._on_position_value_update,
        )
        self.fov_slider_ui_item = dpg.add_slider_int(
            label="FOV",
            min_value=0,
            max_value=180,
            default_value=80,
            callback=self._on_position_value_update,
        )
        pass

    def _on_position_value_update(self) -> None:
        assert self.pitch_slider_ui_item
        assert self.yaw_slider_ui_item
        assert self.fov_slider_ui_item
        self.pitch_slider_value = dpg.get_value(self.pitch_slider_ui_item)
        self.yaw_slider_value = dpg.get_value(self.yaw_slider_ui_item)
        self.fov_slider_value = dpg.get_value(self.fov_slider_ui_item)
        payload: PositionData = PositionData(
            self.pitch_slider_value,
            self.yaw_slider_value,
            self.fov_slider_value,
        )
        self.on_position_value_update(payload)
