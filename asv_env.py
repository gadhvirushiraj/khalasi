"""
Environment for Autonomous Surface Vehicle (ASV) Navigation in different flow fields.
"""

import io
import math
import random

import matplotlib
import numpy as np
import cmasher as cmr
from PIL import Image
import jax.numpy as jnp
import gymnasium as gym
from matplotlib import cm
from gymnasium import spaces

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from jax_vortex_flow_field import VortexFlowField


plt.rcParams["text.color"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["axes.edgecolor"] = "white"
plt.rcParams["axes.titlecolor"] = "white"


class ASVNavEnv(gym.Env):
    """
    ASV Navigation Environment.
    """

    def __init__(
        self,
        training_steps,
        width=300,
        height=100,
        radius=20,
        agent="custom_agent",
        flow_field_type=VortexFlowField,
        headless=True,
    ):
        super(ASVNavEnv, self).__init__()

        self.training_steps = training_steps
        self.WIDTH = width
        self.HEIGHT = height
        self.RADIUS = radius
        self.CIRCLE_X = 3 * self.WIDTH // 4
        self.BORDER_WIDTH = 3
        self.agent_type = agent
        self.spawn_circle_y = self.HEIGHT // 4
        self.goal_circle_y = 3 * self.HEIGHT // 4
        self.headless = headless  # Add a headless attribute
        self.thrust_mag = 0.5

        # only forward thrust
        self.action_space = spaces.Box(
            low=np.array([-self.thrust_mag / 2, -self.thrust_mag / 2]),
            high=np.array([self.thrust_mag, self.thrust_mag]),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )

        self.ORANGE = (255, 165, 0)
        self.GREEN = (0, 255, 0)
        self.BACKGROUND_COLOR = (255, 255, 255)
        self.BLUE = (0, 0, 255)
        self.RED = (255, 0, 0)
        self.BLACK = (0, 0, 0)

        # Initialize flow field
        self.flow_field = flow_field_type(self.WIDTH, self.HEIGHT)

        self.CYLINDER_CENTER_INDEX_X = self.WIDTH // 5
        self.CYLINDER_CENTER_INDEX_Y = self.HEIGHT // 2
        self.CYLINDER_RADIUS_INDICES = self.HEIGHT // 9

        # Initialize screen only if not in headless mode
        if not self.headless:
            matplotlib.use("TkAgg")
            self.fig = plt.figure(figsize=(12, 6), facecolor="black")
            self.gs = GridSpec(
                2, 4, figure=self.fig, width_ratios=[4, 1, 1, 1], height_ratios=[1, 2]
            )
            # add padding between columns
            self.fig.subplots_adjust(wspace=1)
            self.fig.canvas.manager.set_window_title("ASV Navigation Environment")
            self.fig.canvas.manager.window.resizable(False, False)

        self.reset()

    def reset(self, seed=None, options=None):
        """
        Resets the environment to the initial state.
        """

        # init flow field
        self.flow_field.initialize_field()

        # init boat
        self.v = 0
        self.n_steps = 0

        self.thruster1 = 0
        self.thruster2 = 0

        self.x, self.y = self.spawn_boat()
        self.target_x, self.target_y = self.spawn_target((self.x, self.y))
        self.angle_rad = random.uniform(-math.pi, math.pi)

        self.v_x = np.sin(self.angle_rad) * self.v
        self.v_y = np.cos(self.angle_rad) * self.v
        self.flow_field.update_field()

        return self.generate_obs_img(), {}

    def generate_obs_img(self):
        """
        Generates an observation image for the agent with streamlines color-coded based on flow velocity magnitude.
        The agent's dot is color-coded based on its speed, normalized to the flow velocity range.

        Returns:
            np.array: The observation image.
        """

        obs_size, padding = 20, 10
        x_center, y_center = int(self.x), int(self.y)

        # Pad the flow field for boundary handling
        u_field_extended = np.pad(
            self.flow_field.ux, pad_width=padding, mode="constant", constant_values=0
        )
        v_field_extended = np.pad(
            self.flow_field.uy, pad_width=padding, mode="constant", constant_values=0
        )

        # Extract local flow field around the agent
        x_start = max(0, x_center - obs_size // 2 + padding)
        y_start = max(0, y_center - obs_size // 2 + padding)
        x_end = min(u_field_extended.shape[1], x_center + obs_size // 2 + padding)
        y_end = min(v_field_extended.shape[0], y_center + obs_size // 2 + padding)
        u_field = u_field_extended[y_start:y_end, x_start:x_end]
        v_field = v_field_extended[y_start:y_end, x_start:x_end]
        x, y = np.arange(0, u_field.shape[1]), np.arange(0, u_field.shape[0])
        X, Y = np.meshgrid(x, y)

        # Calculate velocity magnitudes
        flow_magnitude = np.sqrt(u_field**2 + v_field**2)

        # Plot the flow field (streamlines with color-coded velocity magnitude)
        fig, ax = plt.subplots(figsize=(2, 2))
        fig.patch.set_facecolor("white")
        streamline = ax.streamplot(
            X,
            Y,
            u_field,
            v_field,
            color=flow_magnitude,  # Use velocity magnitude for coloring
            linewidth=0.8,  # Fixed line width
            density=0.5,
            cmap=cmr.cosmic_r,
            zorder=1,
        )
        ax.set_xlim(0, u_field.shape[1])
        ax.set_ylim(0, u_field.shape[0])
        ax.set_aspect("equal")
        ax.axis("off")

        # Add goal position
        rel_goal_x, rel_goal_y = self.target_x - self.x, self.target_y - self.y
        goal_distance = np.linalg.norm([rel_goal_x, rel_goal_y])
        goal_angle = math.atan2(rel_goal_y, rel_goal_x)

        goal_x_edge = int(u_field.shape[1] // 2 + goal_distance * math.cos(goal_angle))
        goal_y_edge = int(u_field.shape[0] // 2 + goal_distance * math.sin(goal_angle))

        square_size = 2.5
        half_square_size = square_size / 2

        # Adjust to ensure the square is fully inside the image
        goal_x_edge = np.clip(
            goal_x_edge, half_square_size, u_field.shape[1] - half_square_size
        )
        goal_y_edge = np.clip(
            goal_y_edge, half_square_size, u_field.shape[0] - half_square_size
        )

        ax.add_patch(
            plt.Rectangle(
                (goal_x_edge - half_square_size, goal_y_edge - half_square_size),
                square_size,
                square_size,
                color="red",
                zorder=2,
            )
        )

        # Add the agent's position as a color-coded dot
        speed = np.sqrt(self.v_x**2 + self.v_y**2)
        normalized_speed = np.clip(
            speed / 0.25, 0, 1
        )  # Normalize agent's speed to 0-0.3 range
        dot_color = cmr.cosmic_r(
            normalized_speed
        )  # Use the same colormap for the agent

        circle_radius = 1
        circle = plt.Circle(
            (u_field.shape[1] // 2, u_field.shape[0] // 2),
            circle_radius,
            color=dot_color,
            fill=True,
            zorder=2,
        )
        ax.add_patch(circle)

        # Add agent heading direction (red arrow)
        direction_length = 3
        ax.arrow(
            u_field.shape[1] // 2,
            u_field.shape[0] // 2,
            direction_length * math.sin(self.angle_rad),
            direction_length * math.cos(self.angle_rad),
            head_width=1,
            head_length=2,
            fc="red",
            ec="red",
            zorder=3,
        )

        # Save the image to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="PNG", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        plt.close()

        # Convert the buffer image to a numpy array
        pil_image = Image.open(buf).convert("RGB").resize((64, 64))
        image = np.array(pil_image)

        return image

    def step(self, action):

        self.n_steps += 1
        self.thruster1, self.thruster2 = float(action[0]), float(action[1])
        forward_force = self.thruster1 + self.thruster2
        rotation_force = self.thruster2 - self.thruster1

        # Update boat's velocity based on thrusters and the current angle
        self.v_x = forward_force * math.sin(self.angle_rad)
        self.v_y = forward_force * math.cos(self.angle_rad)
        self.angular_v = rotation_force * (math.pi / 20)

        # Update flow field
        self.flow_field.update_field()

        # Determine max index for the flow field (boundary of the field)
        max_x, max_y = self.flow_field.ux.shape[1] - 1, self.flow_field.ux.shape[0] - 1

        # Ensure indices for flow field access are within valid range
        int_x = np.clip(int(self.x), 0, max_x)
        int_y = np.clip(int(self.y), 0, max_y)

        self.pre_x = self.x
        self.pre_y = self.y

        # Update boat's position by adding velocity and flow field's velocity
        self.x += self.v_x + self.flow_field.ux[int_y, int_x]
        self.y += self.v_y + self.flow_field.uy[int_y, int_x]
        self.angle_rad = (self.angle_rad + self.angular_v) % (2 * math.pi)

        done, target_reached = self.check_terminal_state()
        truncated = self.check_truncated()
        reward = self.calculate_reward(done, target_reached, truncated)

        # Add custom metrics to the `info` dictionary
        info = {}
        info["distance_reward"] = self.distance_reward
        info["heading_reward"] = self.heading_reward
        info["target_bonus"] = self.target_bonus
        info["truncation_penalty"] = self.truncation_penalty
        info["thruster_used"] = self.thruster_used
        info["thruster_penalty"] = self.thruster_penalty
        info["goal_reached"] = self.goal_reached

        self.obs_image = self.generate_obs_img()
        if not self.headless:
            self.render()

        print(
            f"Step: {self.n_steps}, Reward: {reward:.2f}, Done: {done}, Truncated: {truncated}, action: {action[0]:.2f}, {action[1]:.2f}"
        )
        return self.obs_image, reward, done, truncated, info

    def calculate_reward(self, done, target_reached, truncated):
        """
        Calculates the reward for the agent based on the current state of the environment.

        Returns:
            float: The reward for the agent.
        """

        # Bonus for reaching the target
        self.target_bonus = 100 if target_reached else 0

        # Calculate the change in distance to the target
        pre_distance = math.sqrt(
            (self.pre_x - self.target_x) ** 2 + (self.pre_y - self.target_y) ** 2
        )
        current_distance = math.sqrt(
            (self.x - self.target_x) ** 2 + (self.y - self.target_y) ** 2
        )

        # Reward for moving closer to the target
        if current_distance < pre_distance:
            self.distance_reward = 5 * (pre_distance - current_distance)
        else:
            self.distance_reward = -5 * (current_distance - pre_distance)

        # Heading reward: encourage alignment with the target direction
        self.heading_reward = 0

        # Penalty if the boat is outside the boundaries of the environment
        self.truncation_penalty = 0

        # Penalize excessive thruster usage
        self.thruster_used = abs(self.thruster1) + abs(self.thruster2)
        self.thruster_penalty = 0

        # Combine all rewards and penalties
        reward = (
            self.distance_reward
            + self.target_bonus
            + self.heading_reward
            + self.truncation_penalty
            + self.thruster_penalty
        )

        return float(reward)

    def check_truncated(self):
        """
        Checks if the episode is truncated due to exceeding the maximum number of steps.

        Returns:
            bool: True if the episode is truncated, False otherwise.
        """
        if self.agent_type == "test":
            return False

        return self.n_steps >= 400  # generally, set at 300 for ealrier experiments

    def check_terminal_state(self):
        """
        Checks if the boat has reached a terminal state.
        Current Terminal States:
        - Boat is outside the boundaries of the environment
        - Boat has collided with the obstacle (center cylinder)
        - Boat has reached the target

        Returns:
            tuple: A tuple containing two boolean values: (terminate, target_reached).
        """
        int_x = int(self.x)
        int_y = int(self.y)
        self.goal_reached = False

        # Check if the boat is outside the boundaries of the environment
        if int_x < 100 or int_x >= self.WIDTH or int_y < 0 or int_y >= self.HEIGHT:
            return True, False

        # Check if the boat has collided with the obstacle (center cylinder)
        if (int_x - self.WIDTH / 4) ** 2 + (int_y - self.HEIGHT / 2) ** 2 <= (
            self.HEIGHT / 4
        ) ** 2:
            return True, False

        # Check if the boat has reached the target
        if (int_x - self.target_x) ** 2 + (int_y - self.target_y) ** 2 <= (
            self.RADIUS / 3
        ) ** 2:
            self.goal_reached = True
            return True, True

        return False, False

    def spawn_boat(self):
        """
        Spawns the boat in a random location before 5/6 of the screen width and below 1/2 of the screen height.

        Returns:
            tuple: The x and y coordinates of the boat.
        """
        random_x = random.uniform(self.WIDTH / 3 + 1, 5 * self.WIDTH / 6)
        random_y = random.uniform(0, self.HEIGHT / 2)
        return random_x, random_y

    def spawn_target(self, boat_position):
        """
        Spawns the target in a random location after the boat's x-coordinate
        and above 1/2 of the screen height.

        Args:
            boat_position (tuple): The x and y coordinates of the boat.

        Returns:
            tuple: The x and y coordinates of the target.
        """
        boat_x, _ = boat_position  # Extract boat's x-coordinate
        random_x = random.uniform(
            boat_x + 1, self.WIDTH
        )  # Ensure target is ahead of the boat
        random_y = random.uniform(self.HEIGHT / 2, self.HEIGHT)
        return random_x, random_y

    def render(
        self,
        return_img=False,
    ):
        """
        Renders the current state of the environment using matplotlib.
        """
        obs_image = self.obs_image
        thruster1 = self.thruster1
        thruster2 = self.thruster2
        angle_rad = self.angle_rad

        if self.headless:
            matplotlib.use("Agg")
            self.fig = plt.figure(figsize=(12, 6), facecolor="black")
            self.gs = GridSpec(
                2, 4, figure=self.fig, width_ratios=[4, 1, 1, 1], height_ratios=[1, 2]
            )
            # add padding between columns
            self.fig.subplots_adjust(wspace=1)
            self.fig.canvas.manager.set_window_title("ASV Navigation Environment")

        X = np.arange(0, self.WIDTH)
        Y = np.arange(0, self.HEIGHT)
        X1, Y1 = np.meshgrid(X, Y, indexing="ij")

        d_u__d_x, d_u__d_y = jnp.gradient(
            self.flow_field.macroscopic_velocities[..., 0]
        )
        d_v__d_x, d_v__d_y = jnp.gradient(
            self.flow_field.macroscopic_velocities[..., 1]
        )
        curl = d_u__d_y - d_v__d_x

        # Vorticity Magnitude Contour Plot
        ax1 = self.fig.add_subplot(self.gs[0, 0])
        contour = ax1.contourf(
            X1,
            Y1,
            curl,
            levels=50,
            cmap=cmr.redshift,
            vmin=-0.02,
            vmax=0.02,
        )
        ax1.set_aspect("equal")
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes(
            "right", size="5%", pad=0.05
        )  # Adjust size and padding
        self.fig.colorbar(contour, cax=cax1, label="Vorticity Magnitude", format="%.3f")
        ax1.add_patch(
            plt.Circle(
                (self.CYLINDER_CENTER_INDEX_X, self.CYLINDER_CENTER_INDEX_Y),
                self.CYLINDER_RADIUS_INDICES,
                color="darkgreen",
            )
        )

        # Quiver Plot (Bottom Left)
        ax2 = self.fig.add_subplot(self.gs[1, :])
        X2, Y2 = np.meshgrid(X, Y, indexing="xy")
        ux = self.flow_field.ux
        uy = self.flow_field.uy

        # Resampling parameters
        x_factor = 10  # Downsample by this factor in X-direction
        y_factor = 5  # Downsample by this factor in Y-direction

        # Downsample grid and average vectors
        new_X = X2[::y_factor, ::x_factor]
        new_Y = Y2[::y_factor, ::x_factor]
        new_ux = ux.reshape(new_Y.shape[0], y_factor, -1, x_factor).mean(axis=(1, 3))
        new_uy = uy.reshape(new_Y.shape[0], y_factor, -1, x_factor).mean(axis=(1, 3))
        magnitude = np.sqrt(new_ux**2 + new_uy**2)

        quiver = ax2.quiver(
            new_X,
            new_Y,
            new_ux,
            new_uy,
            magnitude,
            cmap=cmr.ocean,
            angles="xy",
            scale_units="xy",
            pivot="tail",
            zorder=1,
        )
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes(
            "right", size="5%", pad=0.05
        )  # Adjust size and padding
        self.fig.colorbar(quiver, cax=cax2, label="Velocity Magnitude")
        ax2.set_aspect("equal")
        ax2.add_patch(
            plt.Circle(
                (self.CYLINDER_CENTER_INDEX_X, self.CYLINDER_CENTER_INDEX_Y),
                self.CYLINDER_RADIUS_INDICES,
                color="darkgreen",
                zorder=2,
            )
        )

        # Add target and spawn rectangles
        ax2.add_patch(
            plt.Rectangle(
                (self.WIDTH / 3, self.HEIGHT / 2),
                2 * self.WIDTH / 3,
                self.HEIGHT / 2,
                edgecolor="green",
                fill=False,
                linewidth=1,
            )
        )
        ax2.add_patch(
            plt.Rectangle(
                (self.WIDTH / 3, -5),
                2 * self.WIDTH / 3,
                (self.HEIGHT / 2) + 5 - 2,
                color="orange",
                fill=False,
                linewidth=1,
            )
        )

        # Add boat position and heading
        ax2.plot(self.x, self.y, "ro", markersize=10)
        ax2.arrow(
            self.x,
            self.y,
            5 * math.sin(angle_rad),
            5 * math.cos(angle_rad),
            head_width=2,
            head_length=4,
            linewidth=2,
            fc="r",
            ec="r",
        )

        # Add target position
        ax2.plot(self.target_x, self.target_y, "o", markersize=10, color="yellow")

        # Thruster Bar Plot (Right Column)
        ax3 = self.fig.add_subplot(self.gs[0, 2])
        ax3.bar(
            ["Thruster 1", "Thruster 2"],
            [thruster1, thruster2],
            color=["blue", "orange"],
            width=0.6,
        )
        ax3.set_ylim(-1, 1)
        ax3.axhline(0, color="white", linestyle="--", linewidth=1)
        ax3.set_xticks([0, 1])
        ax3.set_title("Thruster Outputs")
        ax3.set_aspect("equal")
        ax3.set_xticklabels(["TH-L", "TH-R"])

        # Text Box for Info
        info_text = (
            f"$\\mathbf{{Step:}}$ {self.n_steps}\n"
            f"$\\mathbf{{ThrusterL}}$ {thruster1:.2f}\n"
            f"$\\mathbf{{ThrusterR}}$ {thruster2:.2f}\n"
            f"$\\mathbf{{Angle:}}$ {math.degrees(self.angle_rad):.2f}°\n"
            f"$\\mathbf{{Position:}}$ ({self.x:.2f}, {self.y:.2f})\n"
            f"$\\mathbf{{Target:}}$ ({self.target_x:.2f}, {self.target_y:.2f})"
        )
        ax4 = self.fig.add_subplot(self.gs[0, 3])
        ax4.set_ylim(0, 1)
        ax4.set_xlim(-1, 1)
        ax4.axis("off")
        ax4.text(
            0,
            0.5,
            info_text,
            fontsize=12,
            color="white",
            ha="center",
            va="center",
        )

        # Render Observation Image
        ax5 = self.fig.add_subplot(self.gs[0, 1])
        ax5.imshow(obs_image)
        box = ax5.get_position()  # Get current position
        ax5.set_position([box.x0 + 0.04, box.y0, box.width, box.height])
        ax5.axis("off")
        plt.title("Observation Image")

        ax3.set_facecolor("black")
        ax2.set_facecolor("black")

        if return_img:

            buf = io.BytesIO()
            plt.savefig(buf, format="PNG", pad_inches=0)
            buf.seek(0)
            plt.close()
            image = np.array(Image.open(buf).convert("RGB"))

            return image

        plt.tight_layout()
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    def close(self):

        if not self.headless:
            plt.close(self.fig)
