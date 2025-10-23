import dearpygui.dearpygui as dpg
from grid_app.views.main import MainView

dpg.create_context()

main_view = MainView(
    "/Users/illia/Documents/projects/pano-analyzer/panos-new/Ai1Eh2D14bkcmF0rTDtStA.jpg"
)
main_view.setup()

dpg.create_viewport(title="Custom Title")
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary", True)
dpg.start_dearpygui()
dpg.destroy_context()
