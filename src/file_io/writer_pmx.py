# Copyright 2025 rintrint

import struct
from pathlib import Path
from typing import List

from ..core.model import Mesh, Vertex


class BinaryWriter:
    """Helper class for writing binary data (Little Endian)."""

    def __init__(self, path: Path):
        self.f = open(path, "wb")

    def close(self):
        self.f.close()

    def write_bytes(self, data):
        self.f.write(data)

    def write_byte(self, v):
        self.f.write(struct.pack("<B", v))

    def write_sbyte(self, v):
        self.f.write(struct.pack("<b", v))

    def write_short(self, v):
        self.f.write(struct.pack("<h", v))

    def write_int(self, v):
        self.f.write(struct.pack("<i", v))

    def write_float(self, v):
        self.f.write(struct.pack("<f", v))

    def write_vec3(self, v):
        self.f.write(struct.pack("<3f", *v))

    def write_vec2(self, v):
        self.f.write(struct.pack("<2f", *v))

    def write_vec4(self, v):
        self.f.write(struct.pack("<4f", *v))

    def write_text(self, text: str):
        data = text.encode("utf-16-le")
        self.write_int(len(data))
        self.f.write(data)


class PmxWriter:
    """
    Serializes a Mesh IR object into a PMX binary file.
    Handles data adaption (weight normalization, index mapping).
    """

    def __init__(self, mesh: Mesh, output_path: Path):
        self.mesh = mesh
        self.writer = BinaryWriter(output_path)
        self.texture_list: List[str] = []

    def write(self):
        print("[Writer] Writing PMX...")

        # 0. Pre-process: Gather Textures to build Texture List
        self._build_texture_list()

        # 1. Header
        self._write_header()

        # 2. Model Info
        self.writer.write_text(self.mesh.name)  # JP Name
        self.writer.write_text(self.mesh.name)  # EN Name
        self.writer.write_text("Converted by Mod2PMX")  # Comment JP
        self.writer.write_text("Converted by Mod2PMX")  # Comment EN

        # 3. Vertices
        self._write_vertices()

        # 4. Faces (Indices)
        self._write_indices()

        # 5. Textures
        self._write_texture_list()

        # 6. Materials
        self._write_materials()

        # 7. Bones
        # If no bones parsed, generate dummy structure
        self._write_bones()

        # 8. Morphs
        self._write_morphs()

        # 9. Frames
        self._write_frames()

        # 10. Rigids/Joints (Empty)
        self.writer.write_int(0)
        self.writer.write_int(0)

        self.writer.close()
        print("[Writer] Done.")

    def _build_texture_list(self):
        """Collect unique texture paths from materials."""
        unique = set()
        for mat in self.mesh.materials:
            if mat.texture_path:
                unique.add(mat.texture_path)
            else:
                unique.add("tex\\placeholder.png")  # Default
        self.texture_list = sorted(unique)

    def _get_texture_index(self, path: str) -> int:
        if not path:
            path = "tex\\placeholder.png"
        try:
            return self.texture_list.index(path)
        except ValueError:
            return 0

    def _write_header(self):
        self.writer.write_bytes(b"PMX ")
        self.writer.write_float(2.0)
        self.writer.write_byte(8)  # Globals count
        self.writer.write_byte(0)  # UTF16LE
        self.writer.write_byte(2)  # Additional UVs (0=Empty, 1=Color) -> Let's use 2 to be safe (UV1=Empty, UV2=Color)
        self.writer.write_byte(4)  # Vertex Index Size
        self.writer.write_byte(1)  # Texture Index Size
        self.writer.write_byte(1)  # Material Index Size
        self.writer.write_byte(2)  # Bone Index Size
        self.writer.write_byte(2)  # Morph Index Size
        self.writer.write_byte(1)  # Rigid Index Size

    def _write_vertices(self):
        count = len(self.mesh.vertices)
        self.writer.write_int(count)

        # Calculate max bone index for dummy generation later
        self.max_bone_idx = 0

        for v in self.mesh.vertices:
            self.writer.write_vec3(v.position)
            self.writer.write_vec3(v.normal)
            self.writer.write_vec2(v.uv)

            # Add UVs (Fixed 2 for now based on original script logic)
            self.writer.write_vec4((0, 0, 0, 0))
            self.writer.write_vec4(v.color)

            # Weight Handling (BDEF Adapter)
            self._write_vertex_weight(v)

            self.writer.write_float(1.0)  # Edge scale

    def _write_vertex_weight(self, v: Vertex):
        """Adapts arbitrary weights to PMX BDEF1/2/4."""
        # 1. Filter & Shift
        # Note: Mod bone indices usually start at 0.
        # In our generated skeleton, index 0 is "Center" (操作中心),
        # so we map Mod Bone I -> PMX Bone I+1.
        valid = []
        for b_idx, weight in v.weights:
            if weight > 0.0001:
                shifted = b_idx + 1
                valid.append((shifted, weight))
                self.max_bone_idx = max(self.max_bone_idx, shifted)

        # 2. Sort
        valid.sort(key=lambda x: x[1], reverse=True)

        count = len(valid)

        if count == 0:
            # BDEF1 -> Bone 0 (Center)
            self.writer.write_byte(0)
            self.writer.write_short(0)

        elif count == 1:
            # BDEF1
            self.writer.write_byte(0)
            self.writer.write_short(valid[0][0])

        elif count == 2:
            # BDEF2
            self.writer.write_byte(1)
            self.writer.write_short(valid[0][0])
            self.writer.write_short(valid[1][0])
            # Normalize
            total = valid[0][1] + valid[1][1]
            self.writer.write_float(valid[0][1] / total)

        else:
            # BDEF4 (Cap at 4)
            self.writer.write_byte(2)
            bones = [0] * 4
            weights = [0.0] * 4
            limit = min(count, 4)
            total = sum(x[1] for x in valid[:limit])

            for i in range(limit):
                bones[i] = valid[i][0]
                weights[i] = valid[i][1] / total

            for b in bones:
                self.writer.write_short(b)
            for w in weights:
                self.writer.write_float(w)

    def _write_indices(self):
        self.writer.write_int(len(self.mesh.indices))
        for idx in self.mesh.indices:
            self.writer.write_int(idx)

    def _write_texture_list(self):
        self.writer.write_int(len(self.texture_list))
        for path in self.texture_list:
            self.writer.write_text(path)

    def _write_materials(self):
        self.writer.write_int(len(self.mesh.materials))
        for mat in self.mesh.materials:
            self.writer.write_text(mat.name)
            self.writer.write_text(mat.name_en)

            self.writer.write_vec4(mat.diffuse)
            self.writer.write_vec3(mat.specular)
            self.writer.write_float(mat.shininess)
            self.writer.write_vec3(mat.ambient)

            # Flags construction
            # Bit 0: Double sided
            # Bit 1: Ground shadow
            # Bit 2: Self shadow map
            # Bit 3: Self shadow
            # Bit 4: Toon edge
            # Default logic: 1 | 2 | 4 | 8 | 16 = 31
            flag = 0
            flag |= 1  # Double sided
            flag |= 1 << 1
            flag |= 1 << 2
            flag |= 1 << 3
            flag |= 1 << 4
            self.writer.write_byte(flag)

            self.writer.write_vec4(mat.edge_color)
            self.writer.write_float(mat.edge_size)

            # Texture Index
            tex_idx = self._get_texture_index(mat.texture_path)
            self.writer.write_byte(tex_idx)  # Size=1 defined in Header

            # Environment (Sphere) / Toon
            self.writer.write_sbyte(-1)  # Sphere texture index
            self.writer.write_byte(0)    # Sphere mode
            self.writer.write_byte(0)    # Toon sharing flag
            self.writer.write_sbyte(-1)  # Toon texture index

            self.writer.write_text("")  # Memo
            self.writer.write_int(mat.index_count)

    def _write_bones(self):
        # Generate dummy bones based on max_bone_idx used in weights
        # Index 0: Center (操作中心)
        # Index 1..N: Dummy Bones
        total_bones = self.max_bone_idx + 1
        self.writer.write_int(total_bones)

        # 0: Center
        self._write_single_bone("操作中心", "Center", 0, -1, [0, 0, 0], True)

        # 1..N: Dummies
        for i in range(1, total_bones):
            self._write_single_bone(f"Bone_{i}", f"Bone_{i}", i, -1, [0, 0, 0])

    def _write_single_bone(self, name_jp, name_en, idx, parent, pos, is_root=False):
        self.writer.write_text(name_jp)
        self.writer.write_text(name_en)
        self.writer.write_vec3(pos)
        self.writer.write_short(parent)
        self.writer.write_int(0)  # Layer

        # Flags: Visible | Rotate | Move
        # 0x0008 | 0x0002 | 0x0001 = 0x000B -> 11
        # Or standard 30 (0x1E) -> 0001 1110 (Visible, Move, Rotate, Select)
        self.writer.write_byte(30)
        # Flags 2
        self.writer.write_byte(0)

        # Tail Pos (Fixed offset for visual)
        if is_root:
            self.writer.write_vec3([0, 0, 0])  # Connect to nothing? Or logic
        else:
            self.writer.write_vec3([0, 1, 0])

    def _write_morphs(self):
        self.writer.write_int(len(self.mesh.morphs))
        for m in self.mesh.morphs:
            self.writer.write_text(m.name)
            self.writer.write_text(m.name_en)
            self.writer.write_byte(4)  # Panel: Other
            self.writer.write_byte(1)  # Type: Vertex

            self.writer.write_int(len(m.offsets))
            for off in m.offsets:
                self.writer.write_int(off.vertex_index)
                self.writer.write_vec3(off.position_offset)

    def _write_frames(self):
        # 3 Frames: Root, Morph, Bones
        self.writer.write_int(3)

        # 1. Root
        self.writer.write_text("Root")
        self.writer.write_text("Root")
        self.writer.write_byte(1)  # Special
        self.writer.write_int(1)
        self.writer.write_byte(0)  # Target: Bone
        self.writer.write_short(0)  # Bone 0

        # 2. Morph
        self.writer.write_text("表情")
        self.writer.write_text("Expressions")
        self.writer.write_byte(1)
        self.writer.write_int(len(self.mesh.morphs))
        for i in range(len(self.mesh.morphs)):
            self.writer.write_byte(1)  # Target: Morph
            self.writer.write_short(i)

        # 3. Bones (Others)
        self.writer.write_text("骨骼")
        self.writer.write_text("Bones")
        self.writer.write_byte(0)

        # Count = Total bones - 1 (Exclude Center)
        count = max(0, self.max_bone_idx)
        self.writer.write_int(count)
        for i in range(1, self.max_bone_idx + 1):
            self.writer.write_byte(0)
            self.writer.write_short(i)
