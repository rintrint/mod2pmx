# Copyright 2025 rintrint

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Vertex:
    """Represents a single vertex in 3D space."""

    position: List[float]  # [x, y, z]
    normal: List[float]  # [x, y, z]
    uv: List[float]  # [u, v]
    color: List[float]  # [r, g, b, a]

    # List of (bone_index, weight).
    # Allows storing > 4 weights from raw mod data before reduction.
    weights: List[Tuple[int, float]] = field(default_factory=list)


@dataclass
class Material:
    """Represents a material surface properties."""

    name: str
    name_en: str = ""

    # Texture paths (Relative or Absolute)
    # In PMX context, usually only the main texture is used,
    # but we store others (specular, normal map) if parsed from INI.
    texture_path: Optional[str] = None

    # Standard PMX material properties
    diffuse: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])  # RGBA
    specular: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # RGB
    shininess: float = 50.0
    ambient: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])  # RGB

    # Rendering flags (Double-sided, Shadow, etc.)
    flag: int = 0

    edge_color: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    edge_size: float = 1.0

    # The number of indices (faces * 3) this material uses
    index_count: int = 0

    # Store extra attributes from INI (e.g. draw_call_offset) for potential debugging
    properties: Dict[str, any] = field(default_factory=dict)


@dataclass
class Bone:
    """Represents a skeletal bone."""

    name: str
    name_en: str = ""
    index: int = -1
    parent_index: int = -1
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # PMX specific flags (Moveable, Rotatable, Visible, etc.)
    flag: int = 0
    layer: int = 0
    tail_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class MorphOffset:
    """Represents a vertex displacement for a morph."""

    vertex_index: int
    position_offset: List[float]  # [dx, dy, dz]


@dataclass
class Morph:
    """Represents a facial expression or shape key."""

    name: str
    name_en: str = ""
    offsets: List[MorphOffset] = field(default_factory=list)
    category: int = 4  # 0:System, 1:Eyebrow, 2:Eye, 3:Lip, 4:Other


@dataclass
class Mesh:
    """Root container for the 3D model data."""

    name: str
    vertices: List[Vertex] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)
    materials: List[Material] = field(default_factory=list)
    bones: List[Bone] = field(default_factory=list)
    morphs: List[Morph] = field(default_factory=list)

    # Base directory where the model source files are located
    base_dir: Optional[str] = None
