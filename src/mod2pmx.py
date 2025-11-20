# Copyright 2025 rintrint

import math
import os
import re
import struct
import subprocess
import traceback
from pathlib import Path

# ================= CONFIGURATION =================
# Base directories relative to the script location
BASE_MOD_DIR = Path("mod")
BASE_OUTPUT_DIR = Path("pmx")

TEXCONV_PATH = Path("texconv.exe")

# Global Scale setting
SCALE = 0.125
# =================================================


class BinaryWriter:
    def __init__(self, path):
        self.f = open(path, "wb")

    def close(self):
        self.f.close()

    def write_bytes(self, data):
        self.f.write(data)

    def write_byte(self, v):
        self.f.write(struct.pack("<B", v))  # Unsigned Byte

    def write_signed_byte(self, v):
        self.f.write(struct.pack("<b", v))  # Signed Byte

    def write_short(self, v):
        self.f.write(struct.pack("<h", v))  # Signed Short (2 bytes)

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

    def write_text(self, text, encoding="utf-16-le"):
        data = text.encode(encoding)
        self.write_int(len(data))
        self.f.write(data)

    # PMX specific index writers
    # NOTE: Sizes must match the Header definitions in main()
    # Use larger index values here. This may slightly increase file size vs PMXEditor/MMD Tools
    def write_vertex_index(self, idx):
        self.write_int(idx)  # 4 bytes

    def write_bone_index(self, idx):
        self.write_short(idx)  # 2 bytes

    def write_morph_index(self, idx):
        self.write_short(idx)  # 2 bytes

    def write_material_index(self, idx):
        self.write_signed_byte(idx)  # 1 byte


class IniAnalyzer:
    def __init__(self, ini_path):
        self.ini_path = ini_path
        self.components = []
        self.all_textures = set()
        # Store resolved paths and stride for buffers based on INI parsing
        # Structure: {'Index': [{'file': 'path.ib', 'stride': 0}, ...], ...}
        self.buffers = {
            "Index": [],  # List of {'file': name, 'stride': 0}
            "Position": None,
            "Blend": None,
            "TexCoord": None,
            "Color": None,
            "Vector": None,
        }

    def parse(self):
        print(f"INI Parsing: {self.ini_path.name}")
        current_section = ""
        current_res_type = None  # Track which resource we are in (e.g., Position)

        # Regex patterns
        component_pattern = re.compile(r"\[TextureOverride(.*)\]")
        resource_section_pattern = re.compile(r"\[Resource(.*)\]")

        filename_pattern = re.compile(r"filename\s*=\s*(.*)")
        stride_pattern = re.compile(r"stride\s*=\s*(\d+)")
        match_index_pattern = re.compile(r"match_first_index\s*=\s*(\d+)")

        # Texture slots
        texture_slot_pattern = re.compile(r"ps-t\d+\s*=\s*(.*)")

        comp_data = {}

        # Helper to map resource names to internal keys
        def get_res_key(name):
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

        try:
            with open(self.ini_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(";"):
                        continue

                    if line.startswith("["):
                        current_section = line
                        current_res_type = None  # Reset resource tracking

                        # Check for Component
                        comp_match = component_pattern.match(line)
                        if comp_match:
                            c_name = comp_match.group(1).strip()
                            if "VertexLimit" not in c_name and c_name != "IB":
                                comp_data[c_name] = {"name": c_name, "first_index": 0, "count": 0}

                        # Check for Resource Section
                        res_match = resource_section_pattern.match(line)
                        if res_match:
                            res_name = res_match.group(1)
                            current_res_key = get_res_key(res_name)
                            if current_res_key:
                                # Initialize buffer info if not exist or create new entry
                                if current_res_key == "Index":
                                    # Index is a list, we append a new placeholder
                                    self.buffers["Index"].append({"file": None, "stride": 0})
                                    current_res_type = ("Index", -1)  # -1 means last element
                                else:
                                    self.buffers[current_res_key] = {"file": None, "stride": 0}
                                    current_res_type = (current_res_key, None)
                        continue

                    # --- Parsing inside [Resource...] ---
                    if current_res_type:
                        key, idx = current_res_type

                        # Parse Filename
                        file_match = filename_pattern.match(line)
                        if file_match:
                            fname = file_match.group(1).replace('"', "").strip()
                            if idx == -1:
                                self.buffers[key][-1]["file"] = fname
                            else:
                                self.buffers[key]["file"] = fname

                        # Parse Stride
                        stride_match = stride_pattern.match(line)
                        if stride_match:
                            stride_val = int(stride_match.group(1))
                            if idx == -1:
                                self.buffers[key][-1]["stride"] = stride_val
                            else:
                                self.buffers[key]["stride"] = stride_val

                    # --- Parsing inside [TextureOverride...] ---
                    if "TextureOverride" in current_section:
                        c_name = component_pattern.match(current_section).group(1).strip()
                        if c_name in comp_data:
                            idx_match = match_index_pattern.match(line)
                            if idx_match:
                                comp_data[c_name]["first_index"] = int(idx_match.group(1))
                            # Capture textures inside override sections too
                            texture_slot_pattern.match(line)
                            # Note: This just catches direct filenames, handling references to Resources is harder
                            # but the global search below catches actual files usually.

                    # Global texture search
                    if "filename" in line.lower() and ".dds" in line.lower():
                        file_match = filename_pattern.match(line)
                        if file_match:
                            self.all_textures.add(file_match.group(1).replace('"', "").strip())

        except Exception as e:
            print(f"[Error] Failed to read INI: {e}")

        self.components = sorted(comp_data.values(), key=lambda x: x["first_index"])
        print(f"INI Parsed: Found {len(self.components)} components, {len(self.all_textures)} textures.")


# --- Math & Transform Utils ---
def normalize(v):
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length == 0:
        return (0, 0, 0)
    return (v[0] / length, v[1] / length, v[2] / length)


def transform_pos(x, y, z):
    # Blender (Z-up) to PMX (Y-up) with Scale
    # Logic: -x, z, -y
    return (-x * SCALE, z * SCALE, -y * SCALE)


def transform_normal(x, y, z):
    # Normal vector transformation
    return (-x, z, -y)


def transform_morph_delta(x, y, z):
    # Morph offset transformation (same as pos)
    return (-x * SCALE, z * SCALE, -y * SCALE)


# --- Texture Utils ---
def create_placeholder_png(path):
    # 1x1 White Pixel PNG
    png_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xdc\xccY\xe7\x00\x00\x00\x00IEND\xaeB`\x82"
    with open(path, "wb") as f:
        f.write(png_data)


def convert_all_textures(mod_base_path, specific_output_dir, texture_set):
    print("Converting textures...")
    tex_output_dir = Path(specific_output_dir) / "tex"
    tex_output_dir.mkdir(parents=True, exist_ok=True)

    placeholder_path = tex_output_dir / "placeholder.png"
    if not placeholder_path.exists():
        create_placeholder_png(placeholder_path)

    # Index 0 is always placeholder
    converted_list = ["tex\\placeholder.png"]

    if not TEXCONV_PATH.exists():
        print("[Warning] texconv.exe not found. Skipping DDS conversion.")
        # Still return the list assuming user might convert later
        for dds_rel_path in sorted(texture_set):
            file_name = Path(dds_rel_path).stem + ".png"
            converted_list.append(f"tex\\{file_name}")
        return converted_list

    for dds_rel_path in sorted(texture_set):
        clean_dds_path = dds_rel_path.replace('"', "").replace("\\", "/").strip()
        input_path = Path(mod_base_path) / clean_dds_path

        if not input_path.exists():
            # Try to find it case-insensitive or in root
            if (Path(mod_base_path) / Path(clean_dds_path).name).exists():
                input_path = Path(mod_base_path) / Path(clean_dds_path).name
            else:
                continue

        file_name = input_path.stem + ".png"
        output_png_path = tex_output_dir / file_name

        # Run texconv if PNG doesn't exist
        if not output_png_path.exists():
            print(f"Converting: {clean_dds_path}")
            cmd = [str(TEXCONV_PATH), "-ft", "png", "-o", str(tex_output_dir), "-y", str(input_path)]
            try:
                subprocess.run(cmd, capture_output=True, timeout=60, check=True)
            except Exception as e:
                print(f"[Error] Conversion failed for {file_name}: {e}")

        converted_list.append(f"tex\\{file_name}")

    return converted_list


# --- File Finding Logic ---
def find_buffer_file(base_path, ini_filename, default_names):
    """
    Find file using INI name first, then defaults via recursion.
    Attempt to resolve the file path in the following order:
    1. Exact path from INI.
    2. Filename from INI in base_path (ignoring folders).
    3. Recursive search for default_name in base_path (if INI didn't specify).
    """
    if isinstance(default_names, str):
        default_names = [default_names]

    # 1. Try INI path
    if ini_filename:
        clean_path = str(ini_filename).replace("\\", "/").strip('"')
        # Full relative path check
        check_path = base_path / clean_path
        if check_path.exists():
            return check_path

        # Check just filename in root/subdirs
        fname = Path(clean_path).name
        # Simple check in base
        if (base_path / fname).exists():
            return base_path / fname
        # Recursive check for this specific filename
        for f in base_path.rglob(fname):
            return f

    # 2. Fallback: Recursive search for default name (e.g. "Position.buf")
    for name in default_names:
        for f in base_path.rglob(name):
            if f.is_file():
                return f

    # 3. Last resort: Case-insensitive check
    for name in default_names:
        for f in base_path.rglob("*"):
            if f.name.lower() == name.lower():
                return f

    return None


# --- General Buffer Reader with Stride Support ---
def read_buffer_with_stride(file_path, vertex_count, stride, data_format, element_size, default_value):
    """
    Read buffer with generic stride support.

    stride: Total bytes per vertex in the file (e.g. 40).
    data_format: struct format string for the ACTUAL data we want (e.g. '3f').
    element_size: bytes size of the data we want (e.g. 12 for '3f').
    default_value: fallback if file read fails.
    """
    result = []
    if not file_path or not file_path.exists():
        return [default_value] * vertex_count

    with open(file_path, "rb") as f:
        data = f.read()

        # If stride is 0 or not provided, try to guess tightly packed
        if stride == 0:
            stride = element_size

        # Verify file size is sufficient
        if len(data) < vertex_count * stride:
            print(f"[Warning] {file_path.name} is too small for {vertex_count} vertices with stride {stride}.")
            # Fallback: proceed and catch errors, or pad
            pass

        try:
            for i in range(vertex_count):
                offset = i * stride
                if offset + element_size <= len(data):
                    val = struct.unpack_from(f"<{data_format}", data, offset)
                    result.append(val)
                else:
                    result.append(default_value)
        except Exception as e:
            print(f"[Error] Failed reading {file_path.name}: {e}")
            return [default_value] * vertex_count

    return result


def read_morph_data(mesh_dir):
    # ShapeKeys are usually in the same folder as Position.buf
    if not mesh_dir or not mesh_dir.exists():
        return []

    offset_path = mesh_dir / "ShapeKeyOffset.buf"
    id_path = mesh_dir / "ShapeKeyVertexId.buf"
    val_path = mesh_dir / "ShapeKeyVertexOffset.buf"

    if not (offset_path.exists() and id_path.exists() and val_path.exists()):
        # Try finding recursively
        offset_path = find_buffer_file(mesh_dir, None, "ShapeKeyOffset.buf")
        if not offset_path:
            return []
        id_path = offset_path.parent / "ShapeKeyVertexId.buf"
        val_path = offset_path.parent / "ShapeKeyVertexOffset.buf"
        if not (id_path.exists() and val_path.exists()):
            return []

    print("Reading ShapeKey buffers...")
    with open(offset_path, "rb") as f:
        offsets = struct.unpack(f"<{os.path.getsize(offset_path) // 4}I", f.read())
    with open(id_path, "rb") as f:
        vertex_ids = struct.unpack(f"<{os.path.getsize(id_path) // 4}I", f.read())
    with open(val_path, "rb") as f:
        vertex_deltas = struct.unpack(f"<{os.path.getsize(val_path) // 2}e", f.read())

    morphs = []
    shape_key_count = 0

    for i in range(len(offsets)):
        start_idx = offsets[i]
        if i == 0 or start_idx != offsets[i - 1]:
            end_idx = offsets[i + 1] if i + 1 < len(offsets) else len(vertex_ids)
            if end_idx <= start_idx:
                continue
            if start_idx == 0 and i > 0:
                break

            morph_name = f"ShapeKey_{shape_key_count}"
            morph_data = []

            for j in range(start_idx, end_idx):
                if j >= len(vertex_ids):
                    break
                v_id = vertex_ids[j]

                # Stride = 6 (PosDelta XYZ + NormalDelta XYZ)
                # We skip Normal Delta (stride 6, take first 3)
                d_idx = j * 6

                if d_idx + 2 < len(vertex_deltas):
                    dx = vertex_deltas[d_idx]
                    dy = vertex_deltas[d_idx + 1]
                    dz = vertex_deltas[d_idx + 2]

                    # Only record if offset is significant
                    if abs(dx) + abs(dy) + abs(dz) > 0.0001:
                        trans_delta = transform_morph_delta(dx, dy, dz)
                        morph_data.append((v_id, trans_delta))

            if morph_data:
                morphs.append({"name": morph_name, "offsets": morph_data})
                shape_key_count += 1

    print(f"Parsed {len(morphs)} ShapeKeys.")
    return morphs


# --- Core Logic for Single Mod ---
def process_single_mod(ini_path, output_dir):
    """Process a single .ini file and generates a .pmx file in the output directory."""
    mod_base_path = ini_path.parent

    # Determine output filename based on ini filename (e.g. mod.ini -> mod.pmx)
    pmx_filename = ini_path.stem + ".pmx"
    output_file = output_dir / pmx_filename

    # Determine Model Name from Folder Name
    model_name_internal = mod_base_path.name

    print(f"\n>>> Processing: {ini_path}")
    print(f"Target: {output_file}")

    # 1. Parse INI
    analyzer = IniAnalyzer(ini_path)
    analyzer.parse()

    pmx_texture_list = convert_all_textures(mod_base_path, output_dir, analyzer.all_textures)

    # 2. Read Geometry

    # --- Position ---
    pos_info = analyzer.buffers["Position"]  # {'file': '...', 'stride': 40}
    pos_file = pos_info["file"] if pos_info else None
    pos_stride = pos_info["stride"] if pos_info else 0

    pos_path = find_buffer_file(mod_base_path, pos_file, "Position.buf")
    if not pos_path:
        print(f"[Error] Position buffer not found in {mod_base_path}.")
        return

    # Calculate Vertex Count
    # Note: If stride is > 0, use it. If 0, assume packed float3 (12 bytes).
    calc_stride = pos_stride if pos_stride > 0 else 12
    count = os.path.getsize(pos_path) // calc_stride

    print(f"  Detected Stride: {calc_stride}")
    print(f"  Vertex Count: {count}")

    # Read Positions using Stride
    # Format '3f' = 12 bytes
    raw_verts_tuples = read_buffer_with_stride(pos_path, count, calc_stride, "3f", 12, (0, 0, 0))

    # --- Indices ---
    # Handle multiple IBs
    indices = []
    ib_candidates = analyzer.buffers["Index"]  # List of dicts

    # If no IBs in INI, look for defaults
    if not ib_candidates:
        default_ib = find_buffer_file(mod_base_path, None, ["Index.buf", "*.ib"])
        if default_ib:
            # Fake a dict entry
            ib_candidates = [{"file": default_ib.name, "stride": 0}]

    loaded_ib_count = 0
    if ib_candidates:
        for ib_info in ib_candidates:
            ib_name = ib_info["file"]
            ib_path = find_buffer_file(mod_base_path, ib_name, ib_name)
            if ib_path and ib_path.exists():
                # print(f"Reading Index Buffer: {ib_path.name}")
                with open(ib_path, "rb") as f:
                    f_size = os.path.getsize(ib_path)
                    if f_size % 4 == 0:
                        new_indices = struct.unpack(f"<{f_size // 4}I", f.read())
                        indices.extend(new_indices)
                        loaded_ib_count += 1

    if loaded_ib_count == 0:
        print("[Error] No valid Index buffers found.")
        return

    # --- Normals ---
    norm_info = analyzer.buffers["Vector"]
    norm_file = norm_info["file"] if norm_info else None
    norm_stride = norm_info["stride"] if norm_info else 0
    norm_path = find_buffer_file(mod_base_path, norm_file, ["Vector.buf", "Normal.buf"])

    normals = [(0, 1, 0)] * count
    if norm_path:
        # Read raw bytes to handle the specific indexing [i+4]
        with open(norm_path, "rb") as f:
            norm_data = f.read()
            # Guess stride if missing
            if norm_stride == 0:
                norm_stride = len(norm_data) // count

            if norm_stride > 0:
                for i in range(count):
                    offset = i * norm_stride
                    # Legacy logic: indices i*8+4, i*8+5, i*8+6.
                    try:
                        if offset + 7 < len(norm_data):
                            nx = struct.unpack_from("b", norm_data, offset + 4)[0] / 127.0
                            ny = struct.unpack_from("b", norm_data, offset + 5)[0] / 127.0
                            nz = struct.unpack_from("b", norm_data, offset + 6)[0] / 127.0
                            normals[i] = (nx, ny, nz)
                    except Exception:
                        pass

    # --- UVs ---
    uv_info = analyzer.buffers["TexCoord"]
    uv_file = uv_info["file"] if uv_info else None
    uv_stride = uv_info["stride"] if uv_info else 0
    uv_path = find_buffer_file(mod_base_path, uv_file, "TexCoord.buf")

    # UV Format: usually 2 half-floats (4 bytes)
    uvs = read_buffer_with_stride(uv_path, count, uv_stride, "2e", 4, (0, 0))

    # --- Weights (Blend) ---
    blend_info = analyzer.buffers["Blend"]
    blend_file = blend_info["file"] if blend_info else None
    blend_stride = blend_info["stride"] if blend_info else 0
    blend_path = find_buffer_file(mod_base_path, blend_file, "Blend.buf")

    blend_data = []
    max_bone_idx = 0

    if blend_path:
        # Blend format: 4 bytes indices (4B), 4 bytes weights (4B) = 8 bytes total data we want.
        # INI says stride=32. So we read 8 bytes at offset 0, skip 24.
        raw_blends = read_buffer_with_stride(blend_path, count, blend_stride, "4B4B", 8, (0, 0, 0, 0, 0, 0, 0, 0))

        for b_raw in raw_blends:
            # b_raw is tuple of 8 ints
            b_indices = b_raw[0:4]
            b_weights = (b_raw[4] / 255.0, b_raw[5] / 255.0, b_raw[6] / 255.0, b_raw[7] / 255.0)
            blend_data.append((b_indices, b_weights))
            max_bone_idx = max(max_bone_idx, max(b_indices))
    else:
        blend_data = [((0, 0, 0, 0), (1, 0, 0, 0))] * count

    # --- Colors ---
    color_info = analyzer.buffers["Color"]
    color_file = color_info["file"] if color_info else None
    color_stride = color_info["stride"] if color_info else 0
    color_path = find_buffer_file(mod_base_path, color_file, "Color.buf")

    # Color format: 4 bytes (RGBA)
    raw_colors = read_buffer_with_stride(color_path, count, color_stride, "4B", 4, (255, 255, 255, 255))
    colors = [(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0, c[3] / 255.0) for c in raw_colors]

    # --- ShapeKeys ---
    # Assuming they reside in the same folder as Position.buf
    morph_list = read_morph_data(pos_path.parent)

    # 3. Write PMX
    print(f"\nWriting PMX file: {output_file}")
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = BinaryWriter(output_file)

    # [Header]
    writer.write_bytes(b"PMX ")
    writer.write_float(2.0)
    writer.write_byte(8)  # Globals
    writer.write_byte(0)  # Encoding (UTF16LE)
    writer.write_byte(2)  # Additional UVs (UV1=Empty, UV2=Color)
    writer.write_byte(4)  # Vertex Index Size
    writer.write_byte(1)  # Texture Index Size
    writer.write_byte(1)  # Material Index Size
    writer.write_byte(2)  # Bone Index Size
    # Set Morph Index Size to 2 (Short) to handle > 255 morphs
    writer.write_byte(2)  # Morph Index Size
    writer.write_byte(1)  # Rigid Index Size

    # [Model Info]
    writer.write_text(model_name_internal)
    writer.write_text(model_name_internal)
    writer.write_text("converted by mod2pmx")
    writer.write_text("converted by mod2pmx")

    # [Vertices]
    print(f"Writing {count} vertices...")
    writer.write_int(count)

    for i in range(count):
        # 1. Position
        pos = raw_verts_tuples[i]
        px, py, pz = transform_pos(pos[0], pos[1], pos[2])
        writer.write_vec3((px, py, pz))

        # 2. Normal
        nx, ny, nz = normals[i]
        writer.write_vec3(normalize(transform_normal(nx, ny, nz)))

        # 3. UV
        writer.write_vec2(uvs[i])

        # 4. Additional UVs (x2)
        writer.write_vec4((0.0, 0.0, 0.0, 0.0))  # AddUV1 (Empty)
        writer.write_vec4(colors[i])  # AddUV2 (Vertex Color)

        # 5. Weight Optimization (BDEF1 / BDEF2 / BDEF4)
        raw_indices, raw_weights = blend_data[i]

        # Filter valid bones (weight > 0) and SHIFT INDICES
        valid_bones = []
        for j in range(4):
            if raw_weights[j] > 0.0001:
                # +1 for 操作中心
                shifted_idx = raw_indices[j] + 1
                valid_bones.append((shifted_idx, raw_weights[j]))

        # Sort by weight descending
        valid_bones.sort(key=lambda x: x[1], reverse=True)
        b_count = len(valid_bones)

        if b_count == 0:
            # Fallback BDEF1 -> Attach to 操作中心 (Index 0)
            writer.write_byte(0)  # Type 0
            writer.write_bone_index(0)

        elif b_count == 1:
            # BDEF1
            writer.write_byte(0)  # Type 0
            writer.write_bone_index(valid_bones[0][0])

        elif b_count == 2:
            # BDEF2
            writer.write_byte(1)  # Type 1
            b1, w1 = valid_bones[0]
            b2, w2 = valid_bones[1]

            # Normalize w1 (w2 is implicit in PMX)
            w_total = w1 + w2
            writer.write_bone_index(b1)
            writer.write_bone_index(b2)
            writer.write_float(w1 / w_total)

        else:
            # BDEF4 (Type 2) - Handle 3 or 4 bones
            writer.write_byte(2)

            # Prepare buffers
            b_final = [0, 0, 0, 0]
            w_final = [0.0, 0.0, 0.0, 0.0]

            # Take top 4 (if > 4) or all available
            limit = min(b_count, 4)
            w_total = sum(item[1] for item in valid_bones[:limit])

            for k in range(limit):
                b_final[k] = valid_bones[k][0]
                w_final[k] = valid_bones[k][1] / w_total  # Normalize

            for b in b_final:
                writer.write_bone_index(b)
            for w in w_final:
                writer.write_float(w)

        # Edge Scale
        writer.write_float(1.0)

    # [Faces]
    print(f"Writing {len(indices) // 3} faces...")
    writer.write_int(len(indices))
    for idx in indices:
        writer.write_vertex_index(idx)

    # [Textures]
    print(f"Writing {len(pmx_texture_list)} textures...")
    writer.write_int(len(pmx_texture_list))
    for tex_path in pmx_texture_list:
        writer.write_text(tex_path)

    # [Materials]
    print(f"Writing {len(analyzer.components)} materials...")
    comps_to_write = analyzer.components
    if not comps_to_write:
        comps_to_write = [{"name": "Default", "first_index": 0, "count": len(indices)}]

    print(f"Writing {len(comps_to_write)} materials...")
    writer.write_int(len(comps_to_write))

    total_indices_written = 0

    for i, comp in enumerate(comps_to_write):
        c_name = comp["name"]

        # Logic to calculate material face count if not provided
        if i == len(comps_to_write) - 1:
            idx_cnt = len(indices) - total_indices_written
        else:
            # Try to find where the next component starts to calculate count
            next_start = comps_to_write[i + 1]["first_index"]
            current_start = comp["first_index"]
            if next_start > current_start:
                idx_cnt = next_start - current_start
            else:
                # Fallback if logic fails
                idx_cnt = 0

        # Safety check for 0 count on single material
        if idx_cnt <= 0 and len(comps_to_write) == 1:
            idx_cnt = len(indices)

        total_indices_written += idx_cnt

        writer.write_text(c_name)  # JP Name
        writer.write_text(c_name)  # EN Name

        # Diffuse, Specular, Shininess, Ambient
        writer.write_vec4((1, 1, 1, 1))
        writer.write_vec3((0, 0, 0))
        writer.write_float(50)
        writer.write_vec3((0.5, 0.5, 0.5))

        # Material Flags
        is_double_sided = True
        enabled_drop_shadow = True
        enabled_self_shadow_map = True
        enabled_self_shadow = True
        enabled_toon_edge = True

        flags = 0
        flags |= int(is_double_sided)
        flags |= int(enabled_drop_shadow) << 1
        flags |= int(enabled_self_shadow_map) << 2
        flags |= int(enabled_self_shadow) << 3
        flags |= int(enabled_toon_edge) << 4

        writer.write_byte(flags)

        # Edge Color / Size
        writer.write_vec4((0, 0, 0, 1))
        writer.write_float(0.5)

        # Texture Index (0 = placeholder)
        writer.write_signed_byte(0)

        # Sphere / Toon
        writer.write_signed_byte(-1)
        writer.write_byte(0)
        writer.write_byte(0)
        writer.write_signed_byte(-1)

        writer.write_text("")  # Memo
        writer.write_int(idx_cnt)  # Face Vertex Count

    # [Bones]
    # Dummy bones based on max index found in weights
    dummy_bone_count = max_bone_idx + 1
    # +1 for 操作中心
    total_bones = dummy_bone_count + 1

    print(f"Writing {total_bones} bones (1 操作中心 + {dummy_bone_count} dummy)...")
    writer.write_int(total_bones)

    # 1. Write 操作中心 (Index 0)
    writer.write_text("操作中心")
    writer.write_text("view cnt")
    writer.write_vec3((0, 0, 0))  # Pos
    writer.write_bone_index(-1)  # Parent
    writer.write_int(0)  # Layer
    writer.write_byte(30)  # Flags: Move | Rotate | Visible | Controllable
    writer.write_byte(0)  # Flags 2
    writer.write_vec3((0, 0, 0))  # Tail pos

    # 2. Write Original Dummy Bones (Index 1 to N)
    for b_idx in range(dummy_bone_count):
        # +1 for 操作中心
        b_name = f"Bone_{b_idx + 1}"
        writer.write_text(b_name)
        writer.write_text(b_name)
        writer.write_vec3((0, 0, 0))  # Pos
        writer.write_bone_index(-1)  # Parent
        writer.write_int(0)  # Layer
        writer.write_byte(30)  # Flags: Move | Rotate | Visible | Controllable
        writer.write_byte(0)  # Flags 2
        writer.write_vec3((0, 1, 0))  # Tail pos

    # [Morphs]
    print(f"Writing {len(morph_list)} morphs...")
    writer.write_int(len(morph_list))
    for morph in morph_list:
        writer.write_text(morph["name"])
        writer.write_text(morph["name"])
        writer.write_byte(4)  # Category: Other
        writer.write_byte(1)  # Type: Vertex

        offsets = morph["offsets"]
        writer.write_int(len(offsets))
        for v_id, delta in offsets:
            writer.write_vertex_index(v_id)
            writer.write_vec3(delta)

    # [Display Frames]
    print("Writing Display Frames (Root, 表情, その他)...")
    writer.write_int(3)  # Count = 3 Frames

    # Frame 1: Root (Index 0: 操作中心)
    writer.write_text("Root")  # Name JP
    writer.write_text("Root")  # Name EN
    writer.write_byte(1)  # Special Frame
    writer.write_int(1)  # 1 Element
    writer.write_byte(0)  # Type: Bone
    writer.write_bone_index(0)  # Index: 0

    # Frame 2: 表情
    writer.write_text("表情")  # Name JP
    writer.write_text("Morph")  # Name EN
    writer.write_byte(1)  # Special Frame
    writer.write_int(len(morph_list))  # Count = Number of morphs
    for m_idx in range(len(morph_list)):
        writer.write_byte(1)  # Type: Morph
        writer.write_morph_index(m_idx)  # Morph Index

    # Frame 3: その他 (Contains all dummy bones)
    # Indices from 1 to total_bones-1
    writer.write_text("その他")
    writer.write_text("Other")
    writer.write_byte(0)  # Not a Special Frame

    writer.write_int(dummy_bone_count)

    for b_idx in range(1, total_bones):
        writer.write_byte(0)  # Bone
        writer.write_bone_index(b_idx)

    # [Rigid Bodies, Joints] (Empty)
    writer.write_int(0)
    writer.write_int(0)

    writer.close()
    print(f"[Success] Created: {pmx_filename}")


def main():
    print("\n=== Batch Mod to PMX Converter ===")

    if not BASE_MOD_DIR.exists():
        print(f"[Error] '{BASE_MOD_DIR}' folder not found. Please create it and put mods inside.")
        return

    # Recursively find all .ini files
    ini_files = list(BASE_MOD_DIR.rglob("*.ini"))

    if not ini_files:
        print(f"[Info] No .ini files found in '{BASE_MOD_DIR}'.")
        return

    print(f"Found {len(ini_files)} mod configs. Starting batch process...")

    for ini_path in ini_files:
        # Skip if inside a 'Disabled' folder (optional common pattern) or system files
        if "desktop.ini" in ini_path.name.lower() or "disabled" in str(ini_path).lower():
            continue

        # Calculate relative path structure
        # e.g. mod/CharA/SkinB/mod.ini -> CharA/SkinB
        try:
            relative_parent = ini_path.parent.relative_to(BASE_MOD_DIR)
        except ValueError:
            # Should not happen given rglob logic
            continue

        # Mirror structure in output
        target_output_dir = BASE_OUTPUT_DIR / relative_parent

        try:
            process_single_mod(ini_path, target_output_dir)
        except Exception:
            print(f"[Error] Failed processing {ini_path.name}:")
            traceback.print_exc()

    print("\n=== Batch Conversion Complete ===")


if __name__ == "__main__":
    main()
