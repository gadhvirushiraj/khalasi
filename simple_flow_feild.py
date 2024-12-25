"""
Implementation of Simple flow field class.
"""

import numpy as np
from base_flow_field import BaseFlowField


class SimpleFlowField(BaseFlowField):
    """
    Simple flow field class with constant random direction and specified magnitude,
    plus optional added noise. The field does not update over time.
    """

    def __init__(self, width, height, magnitude=0.3, noise_level=0.01):
        super().__init__(width, height)

        # Store parameters
        self.magnitude = magnitude
        self.noise_level = noise_level

        # Initialize the flow field
        self.initialize_field()

    def initialize_field(self):
        """
        Initialize the flow field with a random direction and magnitude, and add noise.
        """

        # Generate a single random angle for the entire field
        angle = np.random.uniform(0, 2 * np.pi)
        self.ux = self.magnitude * np.cos(angle) + self.noise_level * np.random.randn(
            self.height, self.width
        )
        self.uy = self.magnitude * np.sin(angle) + self.noise_level * np.random.randn(
            self.height, self.width
        )

    def update_field(self):
        """
        Update does nothing as per requirement. The field remains static.
        """
        pass
