import open3d as o3d
import os

class SavePointcloudWrapper:
    def __init__(self, save_path: str, width: int = 800, height: int = 600):
        self.save_path = save_path
        self.width = width
        self.height = height
        os.makedirs(self.save_path, exist_ok=True)
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
        self.renderer.scene.set_background([1, 1, 1, 1])  # white background

    def save_pointcloud(self, pcd: o3d.geometry.PointCloud, 
                        camera_positions: dict = None,
                        point_size: float = 3.0,
                        title: str = "Point Cloud"):

        if camera_positions is None:
            # Default camera angles: front, top, side, isometric
            camera_positions = {
                "front": [0, 0, 1],
                "top": [0, 1, 0],
                "side": [1, 0, 0],
                "iso": [1, 1, 1],
            }

        # Set up material for point cloud rendering
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = point_size

        # Clear previous geometry and add the current point cloud
        self.renderer.scene.clear_geometry()
        self.renderer.scene.add_geometry("pcd", pcd, mat)

        # Center of the point cloud
        center = pcd.get_center()

        for view, eye in camera_positions.items():
            self.renderer.setup_camera(60.0, center, eye, [0, 1, 0])  # up vector Y
            img = self.renderer.render_to_image()
            filename = os.path.join(self.save_path, f"{title}_{view}.png")
            o3d.io.write_image(filename, img)
            print(f"Saved view '{title}_{view}' to {filename}")
