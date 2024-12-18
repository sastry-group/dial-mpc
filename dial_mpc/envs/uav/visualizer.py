import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class VisualizerConfig:
    """
    Configuration parameters for the DroneVisualizer.
    """
    # Drone dimensions
    cylinder_radius: float = 0.2
    cylinder_height: float = 0.05
    resolution: int = 20

    # Colors
    drone_color: str = '#00FFAA'      # Electric blue-green
    trail_color: str = '#00FFFF'      # Cyan
    background_color: str = '#1e1e1e' # Dark background
    edge_color: str = 'k'             # Edges for the drone

    # Axis and tick colors
    axis_label_color: str = 'white'
    axis_tick_color: str = 'white'
    axis_line_color: str = 'white'

    # Animation parameters
    interval: int = 20                # Interval between frames in milliseconds
    trail_length: int = 50            # Number of past positions to show in the trail
    num_frames: Optional[int] = None  # Total number of frames (defaults to data length)
    repeat: bool = True               # Whether the animation should repeat

    # Plot settings
    figsize: Tuple[int, int] = (10, 8)
    xlim: Tuple[float, float] = (-6, 6)
    ylim: Tuple[float, float] = (-6, 6)
    zlim: Tuple[float, float] = (-1, 4)
    xlabel: str = 'X'
    ylabel: str = 'Y'
    zlabel: str = 'Z'
    grid: bool = True
    view_elev: int = 30
    view_azim: int = 45
    legend_loc: str = 'upper left'
    font_size: int = 12


class DroneVisualizer:
    """
    A class to visualize and animate the 3D trajectory of a drone.

    Attributes:
        trajectory (np.ndarray): An (N, 12) array containing position and rotation matrices.
        config (VisualizerConfig): Configuration parameters for the visualizer.
    """

    def __init__(self, trajectory: np.ndarray, config: VisualizerConfig = VisualizerConfig()):
        """
        Initializes the DroneVisualizer with trajectory data and configuration.
        """
        if trajectory.ndim != 2 or trajectory.shape[1] <= 12:
            raise ValueError("Trajectory must be of shape (N, 12).")

        self.trajectory = trajectory
        self.config = config
        self.num_frames = config.num_frames if config.num_frames else trajectory.shape[0]
        if self.num_frames > trajectory.shape[0]:
            raise ValueError("num_frames cannot exceed the number of trajectory steps.")

        self.cylinder = self.create_cylinder(
            radius=config.cylinder_radius,
            height=config.cylinder_height,
            resolution=config.resolution
        )

        # Initialize plot elements
        self.fig = None
        self.ax = None
        self.poly_collection = None
        self.trail, = (None,)

    def create_cylinder(self, radius: float, height: float, resolution: int) -> list:
        """
        Creates the vertices of a cylinder aligned along the z-axis.
        """
        theta = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
        z = np.linspace(-height / 2, height / 2, 2)
        theta_grid, z_grid = np.meshgrid(theta, z)

        x_grid = radius * np.cos(theta_grid)
        y_grid = radius * np.sin(theta_grid)

        vertices = []

        for i in range(resolution):
            j = (i + 1) % resolution  # Wrap around to the first vertex

            # Define the four corners of the side face
            face = [
                [x_grid[0, i], y_grid[0, i], z_grid[0, i]],  # Bottom vertex
                [x_grid[0, j], y_grid[0, j], z_grid[0, j]],  # Next bottom vertex
                [x_grid[1, j], y_grid[1, j], z_grid[1, j]],  # Next top vertex
                [x_grid[1, i], y_grid[1, i], z_grid[1, i]]   # Top vertex
            ]
            vertices.append(face)

        return vertices

    def transform_vertices(self, vertices: list, R: np.ndarray, t: np.ndarray) -> list:
        """
        Applies rotation and translation to a list of vertices.
        """
        transformed = []
        for face in vertices:
            transformed_face = []
            for vertex in face:
                v = np.array(vertex)
                v_rot = R @ v       # Apply rotation
                v_trans = v_rot + t  # Apply translation
                transformed_face.append(v_trans)
            transformed.append(transformed_face)
        return transformed

    def initialize_plot(self):
        """
        Initializes the 3D plot with the drone and trail.
        """
        self.fig = plt.figure(figsize=self.config.figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Set plot limits
        self.ax.set_xlim(self.config.xlim)
        self.ax.set_ylim(self.config.ylim)
        self.ax.set_zlim(self.config.zlim)
        
        # Set aspect ratio to be equal
        self.ax.set_aspect('equal')

        # Label axes
        self.ax.set_xlabel(self.config.xlabel, fontsize=self.config.font_size, color=self.config.axis_label_color)
        self.ax.set_ylabel(self.config.ylabel, fontsize=self.config.font_size, color=self.config.axis_label_color)
        self.ax.set_zlabel(self.config.zlabel, fontsize=self.config.font_size, color=self.config.axis_label_color)

        # Improve the visual appearance
        self.ax.grid(self.config.grid)
        self.ax.set_facecolor(self.config.background_color)
        self.fig.patch.set_facecolor(self.config.background_color)

        # Set tick colors
        self.ax.tick_params(axis='x', colors=self.config.axis_tick_color)
        self.ax.tick_params(axis='y', colors=self.config.axis_tick_color)
        self.ax.tick_params(axis='z', colors=self.config.axis_tick_color)

        # Set axis spine colors (for 3D, we can adjust the pane colors)
        self.ax.xaxis.pane.set_edgecolor(self.config.axis_line_color)
        self.ax.yaxis.pane.set_edgecolor(self.config.axis_line_color)
        self.ax.zaxis.pane.set_edgecolor(self.config.axis_line_color)

        # For a more visible axis line, you can adjust pane fill colors:
        self.ax.xaxis.pane.set_facecolor(self.config.background_color)
        self.ax.yaxis.pane.set_facecolor(self.config.background_color)
        self.ax.zaxis.pane.set_facecolor(self.config.background_color)

        # Initialize the drone (cylinder) at the first position
        initial_pos = self.trajectory[0, 0:3]
        initial_R = self.trajectory[0, 3:12].reshape((3, 3))
        transformed_cylinder = self.transform_vertices(self.cylinder, initial_R, initial_pos)
        self.poly_collection = Poly3DCollection(
            transformed_cylinder,
            facecolors=self.config.drone_color,
            edgecolors=self.config.edge_color,
            linewidths=0.5,
            alpha=0.9
        )
        self.ax.add_collection3d(self.poly_collection)

        # Initialize the trailing trajectory
        self.trail, = self.ax.plot(
            [], [], [],
            color=self.config.trail_color,
            linewidth=2,
            label='Trajectory'
        )

        # Optionally, add a legend and set its text color
        legend = self.ax.legend(loc=self.config.legend_loc)
        plt.setp(legend.get_texts(), color=self.config.axis_label_color)

        # Adjust the viewing angle for better visualization
        self.ax.view_init(elev=self.config.view_elev, azim=self.config.view_azim)

    def update(self, frame: int):
        """
        Update function for the animation.
        """
        # Extract current position and rotation
        pos = self.trajectory[frame, 0:3]
        R = self.trajectory[frame, 3:12].reshape((3, 3))

        # Transform the cylinder
        transformed = self.transform_vertices(self.cylinder, R, pos)
        self.poly_collection.set_verts(transformed)

        # Update the trailing path
        trail_length = self.config.trail_length
        if frame < trail_length:
            indices = slice(0, frame + 1)
        else:
            indices = slice(frame - trail_length + 1, frame + 1)

        self.trail.set_data(self.trajectory[indices, 0], self.trajectory[indices, 1])
        self.trail.set_3d_properties(self.trajectory[indices, 2])

        return self.poly_collection, self.trail

    def animate(self):
        """
        Creates and displays the animation.
        """
        self.initialize_plot()

        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=self.num_frames,
            interval=self.config.interval,
            blit=False,
            repeat=self.config.repeat
        )

        plt.show()

    def animate_and_save(self, save_path: str, fps: int = 50):
        """
        Creates and saves the animation to a file.
        """
        self.initialize_plot()

        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=self.num_frames,
            interval=self.config.interval,
            blit=False,
            repeat=self.config.repeat
        )

        # Determine the writer based on file extension
        if save_path.lower().endswith('.mp4'):
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='DroneVisualizer'), bitrate=1800)
        elif save_path.lower().endswith('.gif'):
            Writer = animation.writers['imagemagick']
            writer = Writer(fps=fps)
        else:
            raise ValueError("Unsupported file format. Use .mp4 or .gif.")

        ani.save(save_path, writer=writer)
        print(f'Animation saved to {save_path}')

    @staticmethod
    def generate_dummy_trajectory(N: int) -> np.ndarray:
        """
        Generates a dummy trajectory with changing yaw, pitch, and roll for demonstration purposes.
        """
        trajectory = np.zeros((N, 12))

        # Helical trajectory for position
        t_vals = np.linspace(0, 4 * np.pi, N)
        x = 5 * np.cos(t_vals)
        y = 5 * np.sin(t_vals)
        z = t_vals / (2 * np.pi)
        trajectory[:, 0:3] = np.stack((x, y, z), axis=1)

        # Rotation matrices: varying yaw, pitch, and roll
        for i in range(N):
            yaw = t_vals[i]
            pitch = 0.5 * np.sin(t_vals[i] / 2)
            roll = 0.3 * np.cos(t_vals[i] / 3)

            R_z = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw),  np.cos(yaw), 0],
                [0,             0,          1]
            ])

            R_y = np.array([
                [ np.cos(pitch), 0, np.sin(pitch)],
                [0,              1, 0           ],
                [-np.sin(pitch), 0, np.cos(pitch)]
            ])

            R_x = np.array([
                [1, 0,              0           ],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll),  np.cos(roll)]
            ])

            # Combined rotation matrix: R = R_z * R_y * R_x
            R = R_z @ R_y @ R_x

            trajectory[i, 3:12] = R.flatten()

        return trajectory
