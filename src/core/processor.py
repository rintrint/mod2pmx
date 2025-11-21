# Copyright 2025 rintrint

import math

# Use TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING, Literal, Tuple

if TYPE_CHECKING:
    from .model import Mesh

# Mapping Modes
MappingMode = Literal["UE", "UNITY"]


class ModelProcessor:
    """
    Analyzes and transforms the Mesh data.
    Handles coordinate system conversion (UE/Unity -> MMD) and automatic scaling.
    """

    def __init__(self, mesh: "Mesh"):
        self.mesh = mesh

    def analyze_and_transform(self) -> None:
        """
        Run the main processing pipeline.
        1. Analyze mesh bounds to distinguish between UE (cm) and Unity (m).
        2. Apply specific scaling and coordinate transformation based on the engine.
        """
        if not self.mesh.vertices:
            print("[Processor] Mesh has no vertices. Skipping transform.")
            return

        # 1. Get Bounds to determine "Height" or "Max Span"
        bounds = self._get_bounds()
        span_x = bounds["max_x"] - bounds["min_x"]
        span_y = bounds["max_y"] - bounds["min_y"]
        span_z = bounds["max_z"] - bounds["min_z"]

        # Use the largest dimension as a heuristic for "Height/Scale"
        # This avoids issues if a model is lying down (Z-up vs Y-up)
        max_dimension = max(span_x, span_y, span_z)

        # 2. Detect Engine & Determine Scale/Mapping
        scale, engine_mode = self._detect_engine_and_scale(max_dimension)

        # 3. Apply Transform
        print(f"[Processor] Detected Engine: {engine_mode} (MaxDim={max_dimension:.2f})")
        print(f"[Processor] Applying Transform... Scale: {scale}")
        self._apply_transform(scale, engine_mode)

    def _get_bounds(self) -> dict:
        """Calculate Min/Max for X, Y, and Z axes."""
        xs = [v.position[0] for v in self.mesh.vertices]
        ys = [v.position[1] for v in self.mesh.vertices]
        zs = [v.position[2] for v in self.mesh.vertices]

        return {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys), "min_z": min(zs), "max_z": max(zs)}

    def _detect_engine_and_scale(self, max_dim: float) -> Tuple[float, MappingMode]:
        """
        Heuristic to detect Engine based on Dimensions.

        Logic:
        - UE uses Centimeters. A character is ~160-180 units.
        - Unity uses Meters. A character is ~1.6-1.8 units.
        - MMD target is ~20 units (1 MMD unit ≈ 8cm).

        Decision:
        - If max_dim > 100: Treat as UE (CM). Scale * 0.125.
        - Else: Treat as Unity (Meters). Scale * 12.5.
        """
        if max_dim > 100:
            # UE (cm) -> MMD
            # 160cm * 0.125 = 20 units
            return 0.125, "UE"
        # Unity (m) -> MMD
        # 1.6m * 12.5 = 20 units
        return 12.5, "UNITY"

    def _apply_transform(self, scale: float, mode: MappingMode) -> None:
        """Apply Scale and Coordinate Mapping to all vertices and morphs."""
        for v in self.mesh.vertices:
            raw_pos = v.position
            raw_norm = v.normal

            # --- Position Transform ---
            if mode == "UE":
                # UE (Z-Up) to MMD (Y-Up)
                # Rotate -90 degrees around X-axis logic
                v.position = [
                    -raw_pos[0] * scale,  # -X
                    raw_pos[2] * scale,  # Z becomes Y
                    -raw_pos[1] * scale,  # -Y becomes Z
                ]
            else:  # UNITY
                # Unity (Y-Up, Z-Forward) to MMD (Y-Up, Z-Back)
                # Invert Z-axis
                v.position = [
                    raw_pos[0] * scale,  # X
                    raw_pos[1] * scale,  # Y
                    -raw_pos[2] * scale,  # -Z
                ]

            # --- Normal Transform ---
            # Normals must follow the same rotation/mirroring logic, but WITHOUT scaling.
            if mode == "UE":
                nx = -raw_norm[0]
                ny = raw_norm[2]
                nz = -raw_norm[1]
            else:  # UNITY
                nx = raw_norm[0]
                ny = raw_norm[1]
                nz = -raw_norm[2]  # Mirror Z normal

            # Normalize to ensure precision
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            if length > 0:
                v.normal = [nx / length, ny / length, nz / length]
            else:
                v.normal = [0.0, 0.0, 0.0]

        # --- Morph (Delta) Transform ---
        for morph in self.mesh.morphs:
            for offset in morph.offsets:
                dx, dy, dz = offset.position_offset

                if mode == "UE":
                    offset.position_offset = [-dx * scale, dz * scale, -dy * scale]
                else:  # UNITY
                    offset.position_offset = [dx * scale, dy * scale, -dz * scale]
