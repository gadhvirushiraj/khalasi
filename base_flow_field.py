"""Base class for all different flow fields."""

import numpy as np


class BaseFlowField:
    """Base class for flow fields."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.ux = np.zeros((height, width))  # Placeholder for x-velocity
        self.uy = np.zeros((height, width))  # Placeholder for y-velocity

    def initialize_field(self):
        """Initialize the flow field."""
        raise NotImplementedError("Subclasses should implement this method.")

    def update_field(self):
        """Update the flow field for the next step."""
        raise NotImplementedError("Subclasses should implement this method.")
