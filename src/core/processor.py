# Copyright 2025 rintrint

import math
from typing import Tuple

from .model import Mesh


class ModelProcessor:
    """
    Analyzes and transforms the Mesh data.
    Handles coordinate system conversion and automatic scaling.
    """

    def __init__(self, mesh: Mesh):
        self.mesh = mesh

    def analyze_and_transform(self) -> None:
        """
        Run the main processing pipeline.
        1. Analyze mesh height to determine proper scaling factor.
        2. Apply scaling and coordinate transformation (Rotation/Flip).
        """
        if not self.mesh.vertices:
            print("[Processor] Mesh has no vertices. Skipping transform.")
            return

        # 1. Determine Scale
        min_y, max_y = self._get_height_range()
        height = max_y - min_y

        scale = self._calculate_scale(height)

        # 2. Transform
        print(f"[Processor] Transforming mesh (Scale: {scale})...")
        self._apply_transform(scale)

    def _get_height_range(self) -> Tuple[float, float]:
        """Find the Y-axis bounds of the mesh (in raw coordinates)."""
        # Assuming raw Y is up, or Z is up depending on source.
        # We scan index 1 (Y) for simple heuristic, though source might be Z-up.
        # This is just a rough heuristic for scaling.
        ys = [v.position[1] for v in self.mesh.vertices]
        return min(ys), max(ys)

    def _calculate_scale(self, height: float) -> float:
        """Heuristic to determine if unit conversion is needed."""
        # Standard MMD character height is roughly 10-25 units.
        # If height > 100, it's likely in Centimeters (Game units).
        # If height < 3, it's likely in Meters.

        if height > 100:
            print(f"[Auto-Scale] Detected large model (Raw Height={height:.2f}). Scaling by 0.125.")
            return 0.125
        if height < 3 and height > 0:
            print(f"[Auto-Scale] Detected small model (Raw Height={height:.2f}). Scaling by 12.5.")
            return 12.5
        print(f"[Auto-Scale] Model size looks standard (Raw Height={height:.2f}). Keeping scale 1.0.")
        return 1.0

    def _apply_transform(self, scale: float) -> None:
        """
        Apply Coordinate Transformation:
        Source (Common Game/Blender): Right-Handed? Z-up?
        Target (MMD/PMX): Left-Handed, Y-up

        Logic ported from original script:
        New X = -Old X * Scale
        New Y =  Old Z * Scale  (Swap Y/Z)
        New Z = -Old Y * Scale
        """
        # 1. Transform Vertices
        for v in self.mesh.vertices:
            raw_x, raw_y, raw_z = v.position

            # Position Transform
            v.position = [-raw_x * scale, raw_z * scale, -raw_y * scale]

            # Normal Transform (Rotate only, no scale)
            raw_nx, raw_ny, raw_nz = v.normal
            nx = -raw_nx
            ny = raw_nz
            nz = -raw_ny

            # Normalize ensures precision after flip
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            if length > 0:
                v.normal = [nx / length, ny / length, nz / length]
            else:
                v.normal = [0.0, 0.0, 0.0]

        # 2. Transform Morphs (Deltas)
        for morph in self.mesh.morphs:
            for offset in morph.offsets:
                dx, dy, dz = offset.position_offset
                # Apply same logic as position
                offset.position_offset = [-dx * scale, dz * scale, -dy * scale]
