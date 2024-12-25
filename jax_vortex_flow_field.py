import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import cmasher as cmr
from tqdm import tqdm
from base_flow_field import BaseFlowField
from functools import partial


class VortexFlowField(BaseFlowField):
    """
    Class for genrating Von Karman vortex street flow field.
    """

    def __init__(self, n_points_x, n_points_y):

        # flow field parameters
        self.n_points_x = n_points_x
        self.n_points_y = n_points_y

        # constants
        self.reynolds_number = 80
        self.max_inflow_velocity = 0.15

        # cylinder parameters
        self.cylinder_center_x = n_points_x // 5
        self.cylinder_center_y = n_points_y // 2
        self.cylinder_radius = n_points_y // 9

        # lattice parameters
        self.n_discrete_velocities = 9
        self.lattice_velocities = jnp.array(
            [[0, 1, 0, -1, 0, 1, -1, -1, 1], [0, 0, 1, 0, -1, 1, 1, -1, -1]]
        )
        self.lattice_indices = jnp.arange(self.n_discrete_velocities)
        self.opposite_lattice_indices = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
        self.lattice_weights = jnp.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)

        self.right_velocities = jnp.array([1, 5, 8])
        self.left_velocities = jnp.array([3, 6, 7])
        self.pure_vertical_velocities = jnp.array([0, 2, 4])

        # compute relaxation parameter
        self.kinematic_viscosity = (
            self.max_inflow_velocity * self.cylinder_radius / self.reynolds_number
        )
        self.relaxation_omega = 1.0 / (3.0 * self.kinematic_viscosity + 0.5)

        # generate mesh
        x = jnp.arange(n_points_x)
        y = jnp.arange(n_points_y)
        self.X, self.Y = jnp.meshgrid(x, y, indexing="ij")

        # define obstacle mask
        self.obstacle_mask = (
            jnp.sqrt(
                (self.X - self.cylinder_center_x) ** 2
                + (self.Y - self.cylinder_center_y) ** 2
            )
            < self.cylinder_radius
        )

    def get_density(self, discrete_velocities):
        density = jnp.sum(discrete_velocities, axis=-1)

        return density

    def get_macroscopic_velocities(self, discrete_velocities, density):
        macroscopic_velocities = (
            jnp.einsum(
                "NMQ,dQ->NMd",
                discrete_velocities,
                self.lattice_velocities,
            )
            / density[..., jnp.newaxis]
        )

        return macroscopic_velocities

    def get_equilibrium_discrete_velocities(self, macroscopic_velocities, density):
        projected_discrete_velocities = jnp.einsum(
            "dQ,NMd->NMQ",
            self.lattice_velocities,
            macroscopic_velocities,
        )
        macroscopic_velocity_magnitude = jnp.linalg.norm(
            macroscopic_velocities,
            axis=-1,
            ord=2,
        )
        equilibrium_discrete_velocities = (
            density[..., jnp.newaxis]
            * self.lattice_weights[jnp.newaxis, jnp.newaxis, :]
            * (
                1
                + 3 * projected_discrete_velocities
                + 9 / 2 * projected_discrete_velocities**2
                - 3 / 2 * macroscopic_velocity_magnitude[..., jnp.newaxis] ** 2
            )
        )

        return equilibrium_discrete_velocities

    @partial(jax.jit, static_argnums=0)
    def update(self, discrete_velocities_prev):
        # (1) Prescribe the outflow BC on the right boundary
        discrete_velocities_prev = discrete_velocities_prev.at[
            -1, :, self.left_velocities
        ].set(discrete_velocities_prev[-2, :, self.left_velocities])

        # (2) Macroscopic Velocities
        density_prev = self.get_density(discrete_velocities_prev)
        macroscopic_velocities_prev = self.get_macroscopic_velocities(
            discrete_velocities_prev,
            density_prev,
        )

        # (3) Prescribe Inflow Dirichlet BC using Zou/He scheme
        macroscopic_velocities_prev = macroscopic_velocities_prev.at[0, 1:-1, :].set(
            self.velocity_profile[0, 1:-1, :]
        )
        density_prev = density_prev.at[0, :].set(
            (
                self.get_density(
                    discrete_velocities_prev[0, :, self.pure_vertical_velocities].T
                )
                + 2
                * self.get_density(
                    discrete_velocities_prev[0, :, self.left_velocities].T
                )
            )
            / (1 - macroscopic_velocities_prev[0, :, 0])
        )

        # (4) Compute discrete Equilibria velocities
        equilibrium_discrete_velocities = self.get_equilibrium_discrete_velocities(
            macroscopic_velocities_prev,
            density_prev,
        )

        # (3) Belongs to the Zou/He scheme
        discrete_velocities_prev = discrete_velocities_prev.at[
            0, :, self.right_velocities
        ].set(equilibrium_discrete_velocities[0, :, self.right_velocities])

        # (5) Collide according to BGK
        discrete_velocities_post_collision = (
            discrete_velocities_prev
            - self.relaxation_omega
            * (discrete_velocities_prev - equilibrium_discrete_velocities)
        )

        # (6) Bounce-Back Boundary Conditions to enfore the no-slip
        for i in range(self.n_discrete_velocities):
            discrete_velocities_post_collision = discrete_velocities_post_collision.at[
                self.obstacle_mask, self.lattice_indices[i]
            ].set(
                discrete_velocities_prev[
                    self.obstacle_mask, self.opposite_lattice_indices[i]
                ]
            )

        # (7) Stream alongside lattice velocities
        discrete_velocities_streamed = discrete_velocities_post_collision
        for i in range(self.n_discrete_velocities):
            discrete_velocities_streamed = discrete_velocities_streamed.at[:, :, i].set(
                jnp.roll(
                    jnp.roll(
                        discrete_velocities_post_collision[:, :, i],
                        self.lattice_velocities[0, i],
                        axis=0,
                    ),
                    self.lattice_velocities[1, i],
                    axis=1,
                )
            )

        return discrete_velocities_streamed

    def initialize_field(self):
        flow_states = np.load("vortex_init_states.npy")
        print(f"Flow states shape: {flow_states.shape}")
        random_index = np.random.choice(flow_states.shape[0])
        print(f"Random index: {random_index}")

        # Initialize velocity profile
        self.velocity_profile = jnp.zeros((self.n_points_x, self.n_points_y, 2))
        self.velocity_profile = self.velocity_profile.at[:, :, 0].set(
            self.max_inflow_velocity
        )

        self.discrete_velocities = flow_states[random_index]

    def update_field(self):
        self.discrete_velocities = self.update(self.discrete_velocities)
        density = self.get_density(self.discrete_velocities)
        self.macroscopic_velocities = self.get_macroscopic_velocities(
            self.discrete_velocities, density
        )
        self.ux = self.macroscopic_velocities[..., 0].T
        self.uy = self.macroscopic_velocities[..., 1].T