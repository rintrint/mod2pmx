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
        # Store resolved paths for buffers based on INI parsing
        self.buffers = {
            "Index": None,  # ib
            "Position": None,  # vb (pos)
            "Blend": None,  # vb (blend/weight)
            "TexCoord": None,  # vb (uv)
            "Color": None,  # vb (vertex color)
            "Vector": None,  # vb (normal/tangent)
        }

    def parse(self):
        print(f"INI Parsing: {self.ini_path.name}")
        current_section = ""

        # Regex patterns
        component_pattern = re.compile(r"\[TextureOverrideComponent(\d+)\]")
        # Match resource sections like [ResourcePosition], [ResourceBlend], etc.
        resource_section_pattern = re.compile(r"\[Resource(.*)\]")

        filename_pattern = re.compile(r"filename\s*=\s*(.*)")
        draw_pattern = re.compile(r"drawindexed\s*=\s*(\d+),\s*(\d+),\s*(\d+)")

        # Support direct vb/ib assignment if present (though less common in 3dmigoto for these games)
        # e.g. ib = ... or vb1 = ...
        direct_res_pattern = re.compile(r"^\s*(ib|vb\d+)\s*=\s*(.*)")

        comp_data = {}

        try:
            with open(self.ini_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(";"):
                        continue

                    if line.startswith("["):
                        current_section = line

                        # Check for Component
                        comp_match = component_pattern.match(line)
                        if comp_match:
                            c_id = int(comp_match.group(1))
                            comp_data[c_id] = {"id": c_id, "indices": None}

                        # Check for Resource (Position, Blend, etc.)
                        res_match = resource_section_pattern.match(line)
                        if res_match:
                            # current_section is already set, we will parse filename in the loop
                            pass
                        continue

                    # Parse content based on section
                    if "TextureOverrideComponent" in current_section:
                        draw_match = draw_pattern.match(line)
                        if draw_match:
                            count, start = int(draw_match.group(1)), int(draw_match.group(2))
                            c_id = int(component_pattern.match(current_section).group(1))
                            comp_data[c_id]["indices"] = (start, count)

                    # Check for filename (used in both Textures and Resources)
                    file_match = filename_pattern.match(line)
                    if file_match:
                        fname = file_match.group(1).replace('"', "").strip()

                        # If it's a texture (DDS)
                        if fname.lower().endswith(".dds"):
                            self.all_textures.add(fname)

                        # If it's a buffer in a Resource section
                        elif "Resource" in current_section and ".buf" in fname.lower():
                            res_type = resource_section_pattern.match(current_section).group(1)
                            # Map common names to our internal keys
                            if "Position" in res_type:
                                self.buffers["Position"] = fname
                            elif "Blend" in res_type:
                                self.buffers["Blend"] = fname
                            elif "TexCoord" in res_type:
                                self.buffers["TexCoord"] = fname
                            elif "Index" in res_type:
                                self.buffers["Index"] = fname
                            elif "Color" in res_type:
                                self.buffers["Color"] = fname
                            elif "Vector" in res_type:
                                self.buffers["Vector"] = fname

                    # Check for direct ib/vb assignments (legacy or alternative format)
                    direct_match = direct_res_pattern.match(line)
                    if direct_match:
                        key, val = direct_match.group(1), direct_match.group(2).split(";")[0].strip()
                        if key.lower() == "ib":
                            self.buffers["Index"] = val
                        # Simple heuristic for vbs if not using named sections
                        elif "vb" in key.lower():
                            if "Position" in val:
                                self.buffers["Position"] = val
                            elif "Blend" in val:
                                self.buffers["Blend"] = val
                            elif "TexCoord" in val:
                                self.buffers["TexCoord"] = val

        except Exception as e:
            print(f"[Error] Failed to read INI: {e}")

        # Filter valid components and sort by start index
        valid_comps = [c for c in comp_data.values() if c["indices"] is not None]
        self.components = sorted(valid_comps, key=lambda x: x["indices"][0])
        print(f"INI Parsed: Found {len(self.components)} components, {len(self.all_textures)} textures.")
        # Debug buffer findings
        found_bufs = [k for k, v in self.buffers.items() if v]
        print(f"Buffers found in INI: {found_bufs}")


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


# --- File Finding Logic (XXMI Launcher Style) ---
def find_buffer_file(base_path, ini_path, default_name):
    """
    Attempt to resolve the file path in the following order:
    1. Exact path from INI.
    2. Filename from INI in base_path (ignoring folders).
    3. Recursive search for default_name in base_path (if INI didn't specify).
    """
    # 1. Try INI path
    if ini_path:
        clean_path = str(ini_path).replace("\\", "/").strip('"')
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
    # This mimics XXMI Launcher's scan_directory capability
    for f in base_path.rglob(default_name):
        if f.is_file():
            return f

    # 3. Last resort: Case-insensitive check
    for f in base_path.rglob("*"):
        if f.name.lower() == default_name.lower():
            return f

    return None


# --- Data Reading Utils (Updated to accept Paths) ---
def read_blend_data(blend_path, vertex_count):
    blend_data = []
    max_bone_index = 0

    if blend_path and blend_path.exists():
        print(f"Reading Blend.buf: {blend_path.name}")
        with open(blend_path, "rb") as f:
            data = f.read()
            # Stride 8: 4 bytes indices, 4 bytes weights (0-255)
            if len(data) // 8 != vertex_count:
                print(f"[Warning] Blend.buf size mismatch. Expected {vertex_count} vertices.")

            raw_bytes = struct.unpack(f"<{len(data)}B", data)

            for i in range(0, len(raw_bytes), 8):
                b_indices = (raw_bytes[i], raw_bytes[i + 1], raw_bytes[i + 2], raw_bytes[i + 3])
                b_weights = (raw_bytes[i + 4] / 255.0, raw_bytes[i + 5] / 255.0, raw_bytes[i + 6] / 255.0, raw_bytes[i + 7] / 255.0)
                blend_data.append((b_indices, b_weights))
                max_bone_index = max(max_bone_index, max(b_indices))
    else:
        print("[Warning] Blend.buf not found. Using default weights.")
        blend_data = [((0, 0, 0, 0), (1.0, 0.0, 0.0, 0.0))] * vertex_count

    return blend_data, max_bone_index


def read_color_data(color_path, vertex_count):
    colors = []
    if color_path and color_path.exists():
        print(f"Reading Color.buf: {color_path.name}")
        with open(color_path, "rb") as f:
            data = f.read()
            # RGBA, 1 byte per channel
            if len(data) // 4 == vertex_count:
                raw_bytes = struct.unpack(f"<{len(data)}B", data)
                for i in range(0, len(raw_bytes), 4):
                    r = raw_bytes[i] / 255.0
                    g = raw_bytes[i + 1] / 255.0
                    b = raw_bytes[i + 2] / 255.0
                    a = raw_bytes[i + 3] / 255.0
                    colors.append((r, g, b, a))
            else:
                print("[Warning] Color.buf size mismatch. Filling with white.")
                colors = [(1.0, 1.0, 1.0, 1.0)] * vertex_count
    else:
        colors = [(1.0, 1.0, 1.0, 1.0)] * vertex_count
    return colors


def read_morph_data(mesh_dir):
    # ShapeKeys are usually in the same folder as Position.buf
    if not mesh_dir or not mesh_dir.exists():
        return []

    offset_path = mesh_dir / "ShapeKeyOffset.buf"
    id_path = mesh_dir / "ShapeKeyVertexId.buf"
    val_path = mesh_dir / "ShapeKeyVertexOffset.buf"

    if not (offset_path.exists() and id_path.exists() and val_path.exists()):
        print("[Info] ShapeKey buffers not found (checked relative to Position.buf).")
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

    # Removed hardcoded 'Meshes' directory.
    # We now use dynamic lookup similar to xxmi_launcher logic.

    # Determine output filename based on ini filename (e.g. mod.ini -> mod.pmx)
    pmx_filename = ini_path.stem + ".pmx"
    output_file = output_dir / pmx_filename

    # Determine Model Name from Folder Name
    model_name_internal = mod_base_path.name

    print(f"\n>>> Processing: {ini_path}")
    print(f"Target: {output_file}")

    # 1. Parse INI & Textures (Robust parsing enabled)
    analyzer = IniAnalyzer(ini_path)
    analyzer.parse()

    # Pass output_dir so textures go to pmx/SubFolder/tex
    pmx_texture_list = convert_all_textures(mod_base_path, output_dir, analyzer.all_textures)

    # 2. Read Geometry
    # Use fallback logic if INI paths aren't present
    pos_path = find_buffer_file(mod_base_path, analyzer.buffers["Position"], "Position.buf")
    idx_path = find_buffer_file(mod_base_path, analyzer.buffers["Index"], "Index.buf")

    if not pos_path:
        print(f"[Error] Position buffer (vb) not found in {mod_base_path} or subdirectories.")
        return
    if not idx_path:
        print(f"[Error] Index buffer (ib) not found in {mod_base_path} or subdirectories.")
        return

    print(f"\nReading Geometry from:\n  Pos: {pos_path.name}\n  Idx: {idx_path.name}")

    with open(pos_path, "rb") as f:
        count = os.path.getsize(pos_path) // 12
        raw_verts = struct.unpack(f"<{count * 3}f", f.read())

    with open(idx_path, "rb") as f:
        indices = struct.unpack(f"<{os.path.getsize(idx_path) // 4}I", f.read())

    # Normals (Vector.buf)
    normals = [(0, 1, 0)] * count
    vec_path = find_buffer_file(mod_base_path, analyzer.buffers["Vector"], "Vector.buf")
    if vec_path:
        with open(vec_path, "rb") as f:
            raw_vecs = struct.unpack(f"<{os.path.getsize(vec_path)}b", f.read())
            # Basic check if size matches expectations (stride 8 for normal+tangent is common)
            if len(raw_vecs) == count * 8:
                normals = [(raw_vecs[i + 4] / 127.0, raw_vecs[i + 5] / 127.0, raw_vecs[i + 6] / 127.0) for i in range(0, len(raw_vecs), 8)]

    # UVs (TexCoord.buf)
    uvs = [(0, 0)] * count
    uv_path = find_buffer_file(mod_base_path, analyzer.buffers["TexCoord"], "TexCoord.buf")
    if uv_path:
        with open(uv_path, "rb") as f:
            raw_uvs = struct.unpack(f"<{os.path.getsize(uv_path) // 2}e", f.read())
            uvs = [(raw_uvs[i], raw_uvs[i + 1]) for i in range(0, len(raw_uvs), 8)]

    # Weights & Colors
    blend_path = find_buffer_file(mod_base_path, analyzer.buffers["Blend"], "Blend.buf")
    color_path = find_buffer_file(mod_base_path, analyzer.buffers["Color"], "Color.buf")

    blend_data, max_bone_idx = read_blend_data(blend_path, count)
    colors = read_color_data(color_path, count)

    # ShapeKeys
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
        px, py, pz = transform_pos(raw_verts[i * 3], raw_verts[i * 3 + 1], raw_verts[i * 3 + 2])
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
    writer.write_int(len(analyzer.components))
    for comp in analyzer.components:
        c_id = comp["id"]
        idx_cnt = comp["indices"][1]

        writer.write_text(f"Component_{c_id}")  # JP Name
        writer.write_text(f"Component_{c_id}")  # EN Name

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
    # +1 for 操作中心
    original_bones_count = max_bone_idx + 1
    total_bones = original_bones_count + 1

    print(f"Writing {total_bones} bones (1 操作中心 + {original_bones_count} dummy)...")
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
    for b_idx in range(original_bones_count):
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

    dummy_bone_count = original_bones_count
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
        if "desktop.ini" in ini_path.name.lower():
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
