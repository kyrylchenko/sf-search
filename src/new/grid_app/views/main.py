from grid_app.views.image import ImageView
import dearpygui.dearpygui as dpg
from grid_app.views.side_menu import SideMenuView


class MainView:
    def __init__(self, image_path: str) -> None:
        self.image_view = ImageView(image_path)
        self.side_menu_view = SideMenuView(
            lambda payload: self.image_view.update_image(payload)
        )
        pass

    def setup(self) -> None:
        with dpg.window(tag="Primary"):
            self.image_view.setup()
            self.side_menu_view.setup()
