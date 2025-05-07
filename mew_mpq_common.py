from mew_utils.mew_log import *
import sc2.maps as sc2_map
from pathlib import Path
from PIL import Image
import pympq
import mimetypes
import struct
from enum import Enum

 # Define color codes
colors = {
    "magenta": "\033[35m",
    "bright_cyan": "\033[96m",
    "bright_white": "\033[97m",
    "bright_green": "\033[92m",
    "reset": "\033[0m"
}

line_char = "═"
line = line_char * 12


class MPQType(Enum):
    MAP = "map"
    REPLAY = "replay"

SC2_MAPS_PATH = Path("C:/Program Files (x86)/StarCraft II/Maps")
REPLAY_FOLDER = Path("data/replay_files")

def ansi_color_str(msg, fg):
     return f"{colors[fg]}{msg}{colors['reset']}"

def format_path(directory, depth=0, max_depth=2, max_line_length=80):
    formatted_paths = []
    if max_depth is not None and depth > max_depth:
        return ''

    directory_path = Path(directory)
    if not directory_path.is_dir():
        return ansi_color_str(directory_path, fg='bright_green')

    # Format the current directory path
    indent = "│   " * depth + "├── "
    colored_indent = ansi_color_str(indent, fg='magenta')
    colored_directory = ansi_color_str(directory_path.name, fg='bright_cyan')

    # Check if the length of the directory name exceeds the maximum line length
    if len(colored_indent + colored_directory) > max_line_length:
        colored_directory = "\n" + " " * (len(colored_indent)) + colored_directory

    formatted_paths.append(colored_indent + colored_directory)

    for item in sorted(directory_path.iterdir(), key=lambda x: x.is_file()):
        if item.is_dir() and not item.name.startswith('__') and not item.name.startswith('.'):  # Skip directories starting with '__'
            formatted_paths.append(format_path(item, depth + 1, max_depth, max_line_length))
        elif item.is_file():
            file_indent = "│   " * (depth + 1) + "├── "
            colored_file_indent = ansi_color_str(file_indent, fg='magenta')
            colored_file = ansi_color_str(item.name, fg='bright_green')

            # Check if the length of the file name exceeds the maximum line length
            if len(colored_file_indent + colored_file) > max_line_length:
                colored_file = "\n" + " " * (len(colored_file_indent)) + colored_file

            formatted_paths.append(colored_file_indent + colored_file)

    return '\n'.join(formatted_paths)

def get_all_map_paths():
    return list(SC2_MAPS_PATH.rglob("*.SC2Map")) if SC2_MAPS_PATH.exists() else []

def get_all_map_names():
    return [p.stem for p in get_all_map_paths()]

def get_all_aie_map_names():
    return [p.stem for p in get_all_map_paths() if "AIE" in p.stem]

def get_all_replay_paths():
    return list(REPLAY_FOLDER.glob("*.SC2Replay")) if REPLAY_FOLDER.exists() else []

def get_all_replay_names():
    return [p.stem for p in get_all_replay_paths()]


# ============================

root_data_folder = "data/"

def crop_and_resize_image(input_path, output_path, scale_factor=2):
    """
    Crop an image to remove the black background, resize it, and save as PNG.
    """
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")
            bbox = img.getbbox()
            if bbox:
                cropped_img = img.crop(bbox)
                resized_img = cropped_img.resize(
                    (cropped_img.width * scale_factor, cropped_img.height * scale_factor),
                    Image.LANCZOS,
                )
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                resized_img.save(output_path, format="PNG")
                log_info(
                    f"\n ~ | saved cropped and resized image\n ~ | output_path: {output_path}"
                    f" scale_factor: (x{scale_factor})\n ~ | img resolution: "
                    f"({cropped_img.width}x{cropped_img.height}) => ({cropped_img.width*scale_factor}x{cropped_img.height*scale_factor})"
                )
    except Exception as e:
        log_error(f"Error processing image: {input_path}", e)

def process_binary_file(file_path, output_dir):
    """
    Interpret unknown binary files and save interpreted data.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_txt_path = output_dir / f"{file_path.stem}_utf-8.txt"
        output_txt_path_ascii = output_dir / f"{file_path.stem}_ascii.txt"
        with file_path.open("rb") as f:
            raw_data = f.read()

        with output_txt_path.open("w", encoding="utf-8") as out:
            magic = struct.unpack("<4s", raw_data[:4])[0]
            out.write(f"Magic: {magic.decode('utf-8', errors='ignore')}\n")
            binary_data = raw_data[4:]
            offset = 0
            while offset < len(binary_data) - 4:
                chunk = struct.unpack_from("<4s", binary_data, offset)[0]
                out.write(chunk.decode("utf-8", errors="ignore"))
                offset += 4

        with output_txt_path_ascii.open("w", encoding="ascii") as out:
            magic = struct.unpack("<4s", raw_data[:4])[0]
            out.write(f"Magic: {magic.decode('ascii', errors='ignore')}\n")
            binary_data = raw_data[4:]
            offset = 0
            while offset < len(binary_data) - 4:
                chunk = struct.unpack_from("<4s", binary_data, offset)[0]
                out.write(chunk.decode("ascii", errors="ignore"))
                offset += 4

        log_info(f"\n ~ | saved interpreted binary data to: {output_txt_path} and {output_txt_path_ascii}")
    except Exception as e:
        log_error(f"Error processing binary file: {file_path}", e)


# ===========================

class MPQMapEntry:
    def __init__(self, map_file):
        self.map_file = map_file
        self.extracted_files = {}

    def add_extracted_file(self, mime_type, file_info):
        self.extracted_files.setdefault(mime_type, []).append(file_info)


# only meant for low level binary processing of replays
class MPQReplayEntry:
    def __init__(self, replay_file):
        self.replay_file = replay_file
        self.extracted_files = {}

    def add_extracted_file(self, mime_type, file_info):
        self.extracted_files.setdefault(mime_type, []).append(file_info)

# =========================


def process_archive(archive_path, output_dir, entry_class, mpq_type):
    """
    Process a StarCraft II archive file (map or replay) and extract its contents.

    Args:
        archive_path (str or Path): Path to the .SC2Map or .SC2Replay file.
        output_dir (str or Path): Directory to save the extracted files.
        entry_class: Class to store extracted files (e.g., MPQMapEntry or MPQReplayEntry).
        mpq_type (MPQType): Type of the archive (MAP or REPLAY).
    """
    entry = entry_class(archive_path)
    try:
        with pympq.open_archive(str(archive_path)) as archive:
            output_dir = Path(output_dir) / Path(archive_path).stem
            output_dir.mkdir(parents=True, exist_ok=True)

            file_list = []
            if archive.has_file("(listfile)"):
                listfile_path = output_dir / "listfile.txt"
                archive.extract_file("(listfile)", str(listfile_path))
                with listfile_path.open("rb") as listfile:
                    raw_data = listfile.read()
                    decoded = raw_data.decode("utf-8", errors="replace")
                    file_list = decoded.splitlines()

            for file_name in file_list:
                true_file_name = file_name
                if mpq_type == MPQType.MAP:
                    if file_name in {"DocumentInfo", "Objects", "Triggers", "ComponentList"}:
                        true_file_name += ".xml"
                    if file_name in {"DocumentHeader", "t3SyncTextureInfo"}:
                        true_file_name += ".txt"

                if mpq_type == MPQType.REPLAY:
                    if "tracker" in file_name or "message" in file_name:
                        true_file_name += ".txt"

                output_path = output_dir / true_file_name.replace("\\", "/")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                archive.extract_file(file_name, str(output_path))

                file_size_bytes = output_path.stat().st_size
                file_size_kb = file_size_bytes / 1024
                file_type, _ = mimetypes.guess_type(output_path)
                file_type = file_type if file_type else "???"

                entry.add_extracted_file(file_type, {
                    "file_name": file_name,
                    "path": str(output_path),
                    "size_kb": round(file_size_kb, 2),
                })

                if mpq_type == MPQType.MAP:
                    output_png_path = f"{root_data_folder}extracted_images/{Path(archive_path).stem}/{file_name}.png"
                    if "minimap" in file_name.lower():
                        crop_and_resize_image(output_path, output_png_path, 8)
                    if "image" in file_type:
                        crop_and_resize_image(output_path, output_png_path, 2)
                    if file_type == "???":
                        process_binary_file(output_path, Path(f"{root_data_folder}interpreted_binary_files/{Path(archive_path).stem}/"))

                log_info(
                    f'\n ~ | [{archive_path}] extracted: '
                    f'\n ~ | file_name: "{file_name}"'
                    f'\n ~ | output_path: "{output_path}"'
                    f'\n ~ | size: {file_size_kb:.2f} KB'
                    f'\n ~ | type: {file_type}'
                )
    except Exception as e:
        log_error(f"Error processing archive: {archive_path}", e)

    return entry

def extract_map_mpq_data(map_path, output_dir):
    return process_archive(map_path, output_dir, MPQMapEntry, MPQType.MAP)

# low level handling of replay as binary data
def extract_replay_mpq_data(replay_path, output_dir):
    return process_archive(replay_path, output_dir, MPQReplayEntry, MPQType.REPLAY)


import sc2reader

def format_sc2reader_replay(replay):
    return format_class_attributes(replay)

def sc2reader_load_replay(file_path):
    """Load the SC2Replay file with partial detail."""
    sc2reader.configure(debug=True)
    replay = sc2reader.load_replay(file_path, load_map=False, load_level=3)
    log_info(f"\n ~ | {format_sc2reader_replay(replay)}")
    return replay



def extract_maps_main():
    map_names = get_all_aie_map_names_from_install()
    # map_names = [
    #     "AbyssalReefAIE",
    #     "AutomatonAIE",
    #     "BelShirVestigeAIE",
    #     "DefendersLandingAIE",
    #     "EphemeronAIE",
    #     "Equilibrium513AIE",
    #     "GoldenAura513AIE",
    #     "Gresvan513AIE",
    #     "HardLead513AIE",
    #     "InterloperAIE",
    #     "Oceanborn513AIE",
    #     "SiteDelta513AIE",
    # ]

    output_dir = Path(f"{root_data_folder}extract_map_files")

    maps = {}
    
    for map_str in map_names:
        my_map = sc2_map.get(map_str)
        if my_map:
            maps[map_str] = my_map
            print("=" * 80)
            # Open the MPQ file
            log_info(f"\n ~ | [{map_str}] try map unpack: {my_map.path}")
            try:
                map_registry[map_str] = extract_map_data(my_map.path, output_dir)
            except Exception as e:
                log_error(f" ~ | [{map_str}]: ERROR mpq_map_unpack ", e)

    # Output sorted files
    # Collect output strings
    output_lines = []
    output_lines.append("\n === Sorted Files by Map ===")

    for map_name, mpq_entry in map_registry.items():
        output_lines.append(f"\n ~ | Map: {map_name}")
        for mime_type, files in mpq_entry.extracted_files.items():
            output_lines.append(f"\n ~ | [{map_name}]    MIME Type: {mime_type} (Total: {len(files)})")
            for file_info in files:
                output_lines.append(f"\n ~ | [{map_name}]        - {file_info['file_name']} ({file_info['size_kb']} KB)")

    # Final pass to print all collected output
    log_info("".join(output_lines))


__all__ = [
    # Enums and data structures
    'MPQType',
    'MPQMapEntry',
    'MPQReplayEntry',

    # Color and formatting
    'ansi_color_str',
    'format_path',
    'line_char',
    'line',

    # Path + file utilities
    'SC2_MAPS_PATH',
    'REPLAY_FOLDER',
    'get_all_map_paths',
    'get_all_map_names',
    'get_all_aie_map_names',
    'get_all_replay_paths',
    'get_all_replay_names',

    # Image and binary handling
    'crop_and_resize_image',
    'process_binary_file',

    # Archive processors
    'process_archive',
    'extract_map_mpq_data',
    'extract_replay_mpq_data',

    # Replay utilities
    'sc2reader_load_replay',
    'format_sc2reader_replay',

    # Map batch runner
    'extract_maps_main',

    # Constants
    'root_data_folder',
]
