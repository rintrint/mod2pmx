# Copyright 2025 rintrint

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Set

if TYPE_CHECKING:
    from ..core.model import Mesh


class TextureManager:
    """
    Handle texture processing.

    1. Convert DDS textures to PNG using texconv.exe.
    2. Create a placeholder texture if missing.
    3. Update texture paths in the Mesh object to relative PNG paths.
    """

    def __init__(self, texconv_path: Path, mod_base_dir: Path, output_dir: Path):
        """
        Initialize the TextureManager.

        :param texconv_path: Path to texconv.exe
        :param mod_base_dir: Base directory of the source Mod (for finding absolute paths of DDS)
        :param output_dir: Directory where the PMX will be saved (tex folder will be created here)
        """
        self.texconv_path = texconv_path
        self.mod_base_dir = mod_base_dir
        self.output_dir = output_dir
        self.tex_output_dir = output_dir / "tex"

    def process(self, mesh: "Mesh") -> None:
        """
        Process textures for a given Mesh.

        Main entry point. This modifies the Mesh object in-place (updates material texture paths).
        """
        print("Processing textures...")

        # Ensure output directory exists
        self.tex_output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Ensure placeholder exists (PMX convention often uses a dummy texture at index 0 or for empty slots)
        self._create_placeholder_png(self.tex_output_dir / "placeholder.png")

        # 2. Collect unique texture paths from materials
        # We use a set to avoid processing the same texture multiple times
        unique_dds_paths: Set[str] = set()
        for mat in mesh.materials:
            if mat.texture_path:
                unique_dds_paths.add(mat.texture_path)

        # 3. Batch convert textures and build a mapping (DDS -> PNG)
        # Mapping format: { 'original/path/body.dds' : 'tex/body.png' }
        path_mapping = self._convert_textures(unique_dds_paths)

        # 4. Update Mesh materials with new relative paths
        count = 0
        for mat in mesh.materials:
            if mat.texture_path and mat.texture_path in path_mapping:
                mat.texture_path = path_mapping[mat.texture_path]
                count += 1
            elif not mat.texture_path:
                # Assign placeholder if no texture is defined
                mat.texture_path = "tex\\placeholder.png"

        print(f"Updated {count} material texture references.")

    def _convert_textures(self, dds_paths: Set[str]) -> Dict[str, str]:
        """Convert a set of DDS paths to PNG and return a mapping."""
        mapping = {}

        # Check if texconv exists
        texconv_available = self.texconv_path.exists()
        if not texconv_available:
            print("[Warning] texconv.exe not found. Skipping DDS conversion.")

        for dds_rel_path in sorted(dds_paths):
            # Clean up path string
            clean_path = str(dds_rel_path).replace('"', "").replace("\\", "/").strip()

            # Resolve source path logic (ported from mod2pmx.py)
            input_path = self.mod_base_dir / clean_path

            # Fallback: Try to find file in root if not found in relative path
            if not input_path.exists():
                fallback_path = self.mod_base_dir / Path(clean_path).name
                if fallback_path.exists():
                    input_path = fallback_path
                else:
                    print(f"[Warning] Texture source not found: {clean_path}")
                    # Even if source is missing, we map it to expected PNG path to avoid PMX errors
                    file_name = Path(clean_path).stem + ".png"
                    mapping[dds_rel_path] = f"tex\\{file_name}"
                    continue

            # Determine output PNG path
            file_name = input_path.stem + ".png"
            output_png_path = self.tex_output_dir / file_name

            # Store the relative path for PMX (Windows style backslash)
            relative_pmx_path = f"tex\\{file_name}"
            mapping[dds_rel_path] = relative_pmx_path

            # Skip conversion if PNG already exists to save time
            if output_png_path.exists():
                continue

            if texconv_available:
                self._run_texconv(input_path, output_png_path)
            else:
                # If no texconv, we can't create the PNG, but mapping is already set.
                pass

        return mapping

    def _run_texconv(self, input_path: Path, output_png_path: Path) -> None:
        """Run the external texconv.exe process."""
        # texconv syntax: -ft png -o [output_dir] -y [input_file]
        cmd = [
            str(self.texconv_path),
            "-ft",
            "png",
            "-o",
            str(self.tex_output_dir),
            "-y",
            str(input_path),
        ]

        try:
            # print(f"Converting: {input_path.name}")
            subprocess.run(cmd, capture_output=True, timeout=60, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[Error] texconv failed for {input_path.name}: {e}")
        except Exception as e:
            print(f"[Error] Unexpected error converting {input_path.name}: {e}")

    def _create_placeholder_png(self, path: Path) -> None:
        """
        Create a 1x1 white pixel PNG if it doesn't exist.

        Used as a fallback for missing textures.
        """
        if path.exists():
            return

        # 1x1 White Pixel PNG binary data
        png_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xdc\xccY\xe7\x00\x00\x00\x00IEND\xaeB`\x82"

        try:
            with open(path, "wb") as f:
                f.write(png_data)
        except Exception as e:
            print(f"[Error] Failed to create placeholder PNG: {e}")
