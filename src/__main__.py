# Copyright 2025 rintrint

import traceback
from pathlib import Path

# Module Imports
# Note: Assumes src/file_io/reader_mod.py and src/file_io/writer_pmx.py exist
from .core.processor import ModelProcessor
from .file_io.reader_mod import ModReader
from .file_io.texture_manager import TextureManager
from .file_io.writer_pmx import PmxWriter

# ================= CONFIGURATION =================
BASE_MOD_DIR = Path("mod")
BASE_OUTPUT_DIR = Path("pmx")
TEXCONV_PATH = Path("texconv.exe")
# =================================================


def process_single_file(ini_path: Path) -> None:
    """Pipeline for processing a single mod file."""
    # 1. Determine Output Path
    # Mirror the folder structure from 'mod/' to 'pmx/'
    try:
        relative_path = ini_path.parent.relative_to(BASE_MOD_DIR)
    except ValueError:
        # Fallback if ini is not strictly under BASE_MOD_DIR
        relative_path = ini_path.parent.name

    output_dir = BASE_OUTPUT_DIR / relative_path
    output_file = output_dir / (ini_path.stem + ".pmx")

    print(f"\n>>> Processing: {ini_path}")
    print(f"    Target: {output_file}")

    # 2. READ (Mod -> Mesh IR)
    # Parse INI, buffers, and build the generic Mesh object.
    reader = ModReader(ini_path)
    mesh = reader.read()

    if not mesh:
        print("[Error] Failed to read mesh data.")
        return

    # 3. PROCESS (Geometry & Scale)
    # Analyze height, determine scale, and convert coordinates to MMD system.
    processor = ModelProcessor(mesh)
    processor.analyze_and_transform()

    # 4. ASSETS (Texture Conversion)
    # Convert DDS to PNG and update texture paths inside the Mesh.
    tex_manager = TextureManager(TEXCONV_PATH, ini_path.parent, output_dir)
    tex_manager.process(mesh)

    # 5. WRITE (Mesh IR -> PMX)
    # Serialize the processed Mesh into PMX binary format.
    writer = PmxWriter(mesh, output_file)
    writer.write()

    print(f"[Success] {output_file.name} created.")


def main():
    print("\n=== Mod to PMX Converter (Refactored) ===")

    if not BASE_MOD_DIR.exists():
        print(f"[Error] '{BASE_MOD_DIR}' folder not found. Please create it and put mods inside.")
        return

    # Recursively find all .ini files
    ini_files = list(BASE_MOD_DIR.rglob("*.ini"))

    if not ini_files:
        print(f"[Info] No .ini files found in '{BASE_MOD_DIR}'.")
        return

    print(f"Found {len(ini_files)} mod configs. Starting batch process...")

    success_count = 0
    fail_count = 0

    for ini_path in ini_files:
        # Filter out common non-mod INI files or disabled folders
        if ini_path.name.lower() == "desktop.ini":
            continue
        if "disabled" in str(ini_path).lower():
            continue

        try:
            process_single_file(ini_path)
            success_count += 1
        except Exception:
            print(f"[Fail] Error processing {ini_path.name}:")
            traceback.print_exc()
            fail_count += 1

    print("\n========================================")
    print(f"Batch Complete. Success: {success_count}, Failed: {fail_count}")
    print("========================================")


if __name__ == "__main__":
    main()
