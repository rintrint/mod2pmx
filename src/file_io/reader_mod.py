# Copyright 2025 rintrint

import os
import re
import struct
from pathlib import Path
from typing import List, Optional, Tuple

# Import IR models
# Note: Adjust the import path if your structure is different
from ..core.model import Material, Mesh, Morph, MorphOffset, Vertex


class ModReader:
    """Parses mod .ini files and binary buffers to construct a generic Mesh IR."""

    def __init__(self, ini_path: Path):
        self.ini_path = ini_path
        self.base_dir = ini_path.parent
        # Internal state to track buffer configs from INI
        self.buffer_configs = {
            "Index": [],
            "Position": None,
            "Blend": None,
            "TexCoord": None,
            "Color": None,
            "Vector": None,  # Normal
        }
        self.materials_info = []  # Stores raw material config

    def read(self) -> Optional[Mesh]:
        print(f"[Reader] Parsing INI: {self.ini_path.name}")

        # 1. Parse INI to understand file structure
        self._parse_ini()

        # 2. Initialize Mesh
        mesh = Mesh(name=self.base_dir.name, base_dir=str(self.base_dir))

        # 3. Read Buffers
        # We need to read raw lists first, then zip them into Vertex objects
        vertex_count, raw_positions = self._read_positions()

        if vertex_count == 0:
            print("[Reader] No vertices found. Aborting.")
            return None

        raw_normals = self._read_normals(vertex_count)
        raw_uvs = self._read_uvs(vertex_count)
        raw_colors = self._read_colors(vertex_count)
        raw_blends = self._read_blends(vertex_count)

        # 4. Construct Vertices (Zip data)
        print(f"[Reader] Constructing {vertex_count} vertices...")
        for i in range(vertex_count):
            v = Vertex(
                position=list(raw_positions[i]),
                normal=list(raw_normals[i]),
                uv=list(raw_uvs[i]),
                color=list(raw_colors[i]),
                weights=raw_blends[i],  # List of (bone_idx, weight)
            )
            mesh.vertices.append(v)

        # 5. Read Indices
        mesh.indices = self._read_indices()

        # 6. Read Materials
        # If no materials found in INI, create a default one
        if not self.materials_info:
            mesh.materials.append(Material(name="Default", index_count=len(mesh.indices)))
        else:
            total_indices_written = 0
            for i, info in enumerate(self.materials_info):
                mat = Material(name=info["name"], name_en=info["name"], texture_path=info.get("texture_file", None))

                # Calculate Index Count for this material
                # Logic ported from original mod2pmx
                if i == len(self.materials_info) - 1:
                    count = len(mesh.indices) - total_indices_written
                else:
                    next_start = self.materials_info[i + 1]["first_index"]
                    current_start = info["first_index"]
                    count = max(0, next_start - current_start)

                if count <= 0 and len(self.materials_info) == 1:
                    count = len(mesh.indices)

                mat.index_count = count
                total_indices_written += count
                mesh.materials.append(mat)

        # 7. Read Morphs (ShapeKeys)
        # Usually located in the same folder as Position buffer
        pos_buf_path = self._find_buffer_path("Position", "Position.buf")
        if pos_buf_path:
            mesh.morphs = self._read_morphs(pos_buf_path.parent)

        return mesh

    def _parse_ini(self):
        """Parse .ini file to extract buffer info and material/component info."""
        current_section = ""
        current_res_type = None

        # Regex
        comp_pattern = re.compile(r"\[TextureOverride(.*)\]")
        res_pattern = re.compile(r"\[Resource(.*)\]")
        filename_pattern = re.compile(r"filename\s*=\s*(.*)")
        stride_pattern = re.compile(r"stride\s*=\s*(\d+)")
        match_idx_pattern = re.compile(r"match_first_index\s*=\s*(\d+)")
        # Simple texture grabber for TextureOverride
        re.compile(r"ps-t\d+\s*=\s*Resource(.*)")

        comp_map = {}  # Temporary storage for components

        try:
            with open(self.ini_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(";"):
                        continue

                    if line.startswith("["):
                        current_section = line
                        current_res_type = None

                        # [TextureOverride...] -> Material
                        comp_match = comp_pattern.match(line)
                        if comp_match:
                            c_name = comp_match.group(1).strip()
                            if "VertexLimit" not in c_name and c_name != "IB":
                                comp_map[c_name] = {"name": c_name, "first_index": 0, "texture_res": None}

                        # [Resource...] -> Buffer
                        res_match = res_pattern.match(line)
                        if res_match:
                            res_name = res_match.group(1)
                            key = self._get_resource_key(res_name)
                            if key:
                                if key == "Index":
                                    self.buffer_configs["Index"].append({"file": None, "stride": 0})
                                    current_res_type = ("Index", -1)
                                else:
                                    self.buffer_configs[key] = {"file": None, "stride": 0}
                                    current_res_type = (key, None)
                        continue

                    # Inside [Resource...]
                    if current_res_type:
                        key, idx = current_res_type
                        f_match = filename_pattern.match(line)
                        if f_match:
                            val = f_match.group(1).replace('"', "").strip()
                            if idx == -1:
                                self.buffer_configs[key][-1]["file"] = val
                            else:
                                self.buffer_configs[key]["file"] = val

                        s_match = stride_pattern.match(line)
                        if s_match:
                            val = int(s_match.group(1))
                            if idx == -1:
                                self.buffer_configs[key][-1]["stride"] = val
                            else:
                                self.buffer_configs[key]["stride"] = val

                    # Inside [TextureOverride...]
                    if "TextureOverride" in current_section:
                        c_match = comp_pattern.match(current_section)
                        if c_match:
                            c_name = c_match.group(1).strip()
                            if c_name in comp_map:
                                idx_match = match_idx_pattern.match(line)
                                if idx_match:
                                    comp_map[c_name]["first_index"] = int(idx_match.group(1))

                                # Try to capture texture filename directly
                                if "filename" in line.lower() and ".dds" in line.lower():
                                    f_match = filename_pattern.match(line)
                                    if f_match:
                                        comp_map[c_name]["texture_file"] = f_match.group(1).replace('"', "").strip()

        except Exception as e:
            print(f"[Reader] INI Parse Error: {e}")

        # Sort materials by index
        self.materials_info = sorted(comp_map.values(), key=lambda x: x["first_index"])

    def _get_resource_key(self, name):
        if "Position" in name:
            return "Position"
        if "Blend" in name:
            return "Blend"
        if "TexCoord" in name:
            return "TexCoord"
        if "Color" in name:
            return "Color"
        if "Vector" in name:
            return "Vector"
        if "IB" in name or "Index" in name:
            return "Index"
        return None

    def _read_positions(self) -> Tuple[int, List[Tuple[float, float, float]]]:
        conf = self.buffer_configs["Position"]
        file_name = conf["file"] if conf else None
        stride = conf["stride"] if conf else 0

        path = self._find_buffer_file(file_name, "Position.buf")
        if not path:
            return 0, []

        calc_stride = stride if stride > 0 else 12  # Default float3
        count = os.path.getsize(path) // calc_stride

        data = self._read_buffer_data(path, count, calc_stride, "3f", 12, (0, 0, 0))
        return count, data

    def _read_normals(self, count) -> List[Tuple[float, float, float]]:
        conf = self.buffer_configs["Vector"]
        file_name = conf["file"] if conf else None
        stride = conf["stride"] if conf else 0

        path = self._find_buffer_file(file_name, ["Vector.buf", "Normal.buf"])
        if not path:
            return [(0, 1, 0)] * count

        # Special handling for Normal byte storage logic from original script
        # Indices [i+4], [i+5], [i+6]
        result = []
        with open(path, "rb") as f:
            raw = f.read()
            if stride == 0:
                stride = len(raw) // count
            if stride == 0:
                stride = 8  # Fallback

            for i in range(count):
                offset = i * stride
                try:
                    if offset + 7 < len(raw):
                        nx = struct.unpack_from("b", raw, offset + 4)[0] / 127.0
                        ny = struct.unpack_from("b", raw, offset + 5)[0] / 127.0
                        nz = struct.unpack_from("b", raw, offset + 6)[0] / 127.0
                        result.append((nx, ny, nz))
                    else:
                        result.append((0, 1, 0))
                except Exception:
                    result.append((0, 1, 0))
        return result

    def _read_uvs(self, count) -> List[Tuple[float, float]]:
        conf = self.buffer_configs["TexCoord"]
        file_name = conf["file"] if conf else None
        stride = conf["stride"] if conf else 0
        path = self._find_buffer_file(file_name, "TexCoord.buf")

        # UV is usually 2 half-floats (2e) = 4 bytes
        return self._read_buffer_data(path, count, stride, "2e", 4, (0, 0))

    def _read_colors(self, count) -> List[Tuple[float, float, float, float]]:
        conf = self.buffer_configs["Color"]
        file_name = conf["file"] if conf else None
        stride = conf["stride"] if conf else 0
        path = self._find_buffer_file(file_name, "Color.buf")

        # Color is 4 bytes (RGBA)
        raw = self._read_buffer_data(path, count, stride, "4B", 4, (255, 255, 255, 255))
        return [(r / 255.0, g / 255.0, b / 255.0, a / 255.0) for r, g, b, a in raw]

    def _read_blends(self, count) -> List[List[Tuple[int, float]]]:
        """
        Read bone indices and weights.
        Returns: List of List[(bone_index, weight)]
        """
        conf = self.buffer_configs["Blend"]
        file_name = conf["file"] if conf else None
        stride = conf["stride"] if conf else 0
        path = self._find_buffer_file(file_name, "Blend.buf")

        result = []
        if path:
            # 4 Indices (Bytes) + 4 Weights (Bytes) = 8 bytes
            # INI usually says stride 32, but we only need the first 8
            raw_data = self._read_buffer_data(path, count, stride, "4B4B", 8, (0, 0, 0, 0, 0, 0, 0, 0))

            for item in raw_data:
                indices = item[0:4]
                # Convert byte weights (0-255) to float (0.0-1.0)
                weights = [w / 255.0 for w in item[4:8]]

                # Pack into tuple list, filter zero weights later in Writer or here?
                # Better keep raw data here.
                bone_data = [(indices[i], weights[i]) for i in range(4) if weights[i] > 0]
                result.append(bone_data)
        else:
            # No blend buffer, attach to root or handle in writer
            result = [[] for _ in range(count)]

        return result

    def _read_indices(self) -> List[int]:
        indices = []
        ib_configs = self.buffer_configs["Index"]

        # Fallback default search
        if not ib_configs:
            default_ib = self._find_buffer_file(None, ["Index.buf", "*.ib"])
            if default_ib:
                ib_configs = [{"file": default_ib.name}]

        for conf in ib_configs:
            path = self._find_buffer_file(conf["file"], conf["file"])
            if path and path.exists():
                f_size = os.path.getsize(path)
                if f_size % 4 == 0:
                    with open(path, "rb") as f:
                        indices.extend(struct.unpack(f"<{f_size // 4}I", f.read()))
        return indices

    def _read_morphs(self, search_dir: Path) -> List[Morph]:
        """Read shape keys from auxiliary buffers."""
        offset_path = self._find_buffer_file(None, "ShapeKeyOffset.buf", base=search_dir)
        if not offset_path:
            return []

        id_path = offset_path.parent / "ShapeKeyVertexId.buf"
        val_path = offset_path.parent / "ShapeKeyVertexOffset.buf"

        if not (id_path.exists() and val_path.exists()):
            return []

        print("[Reader] Parsing Morph Buffers...")
        with open(offset_path, "rb") as f:
            offsets = struct.unpack(f"<{os.path.getsize(offset_path) // 4}I", f.read())
        with open(id_path, "rb") as f:
            vertex_ids = struct.unpack(f"<{os.path.getsize(id_path) // 4}I", f.read())
        with open(val_path, "rb") as f:
            # Half-float format 'e'
            vertex_deltas = struct.unpack(f"<{os.path.getsize(val_path) // 2}e", f.read())

        morphs = []
        count = 0

        for i in range(len(offsets)):
            start = offsets[i]
            # Determine end
            if i == 0 or start != offsets[i - 1]:
                end = offsets[i + 1] if i + 1 < len(offsets) else len(vertex_ids)
                if end <= start:
                    continue
                if start == 0 and i > 0:
                    break  # Optimization: usually stop if looping back

                m_name = f"ShapeKey_{count}"
                m_offsets = []

                for j in range(start, end):
                    if j >= len(vertex_ids):
                        break
                    vid = vertex_ids[j]
                    # Stride 6: PosDelta(3) + NormalDelta(3)
                    d_idx = j * 6
                    if d_idx + 2 < len(vertex_deltas):
                        dx = vertex_deltas[d_idx]
                        dy = vertex_deltas[d_idx + 1]
                        dz = vertex_deltas[d_idx + 2]
                        if abs(dx) + abs(dy) + abs(dz) > 0.0001:
                            m_offsets.append(MorphOffset(vid, [dx, dy, dz]))

                if m_offsets:
                    morphs.append(Morph(name=m_name, offsets=m_offsets))
                    count += 1

        return morphs

    # --- Utility Helpers ---

    def _find_buffer_path(self, file_name, default_names) -> Optional[Path]:
        return self._find_buffer_file(file_name, default_names)

    def _find_buffer_file(self, file_name, default_names, base=None) -> Optional[Path]:
        search_base = base if base else self.base_dir
        if isinstance(default_names, str):
            default_names = [default_names]

        # 1. Explicit INI path
        if file_name:
            clean = str(file_name).replace("\\", "/").strip('"')
            check = search_base / clean
            if check.exists():
                return check
            # Check just filename in root/subdirs
            fname = Path(clean).name
            for f in search_base.rglob(fname):
                return f

        # 2. Defaults
        for name in default_names:
            for f in search_base.rglob(name):
                return f
            # Case insensitive
            for f in search_base.rglob("*"):
                if f.name.lower() == name.lower():
                    return f
        return None

    def _read_buffer_data(self, path, count, stride, fmt, elem_size, default_val):
        result = []
        if not path or not path.exists():
            return [default_val] * count

        with open(path, "rb") as f:
            data = f.read()
            if stride == 0:
                stride = elem_size

            try:
                for i in range(count):
                    offset = i * stride
                    if offset + elem_size <= len(data):
                        result.append(struct.unpack_from(f"<{fmt}", data, offset))
                    else:
                        result.append(default_val)
            except Exception:
                pass

        # Fill rest if short
        if len(result) < count:
            result.extend([default_val] * (count - len(result)))
        return result
