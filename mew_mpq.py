import sc2.maps as sc2_map
from mew_utils.mew_log import *
from mew_mpq_common import *

from pathlib import Path

# ============================

root_data_folder = "data/"
map_registry = {}
replay_registry = {}


def extract_map_data(map_path, output_dir):
    return process_archive(map_path, output_dir, MPQMapEntry, MPQType.MAP)


def extract_replay_data(replay_path, output_dir):
    # replay = sc2reader_load_replay(replay_path)
    # events = extract_events(replay)

    # for time_sec, text in events:
    #     log_info(f"[{format_time(time_sec)}] {text}")
    # return ""
    return process_archive(replay_path, output_dir, MPQReplayEntry, MPQType.REPLAY)

def extract_replay_data_w_sc2reader(replay_path, output_dir):
    replay = sc2reader_load_replay(replay_path)
    events = extract_events(replay)
    event_lines = []
    for mew_event in events:
        print(f"\n ~ | {mew_event}")
        print("_"*80)
        event_lines.append(f"{mew_event}")
    return "\n".join(event_lines)

# ===================================

def extract_maps_main():
    map_names = get_all_aie_map_names()
    output_dir = Path(f"{root_data_folder}extract_map_files")

    for map_str in map_names:
        my_map = sc2_map.get(map_str)
        if my_map:
            print("=" * 80)
            log_info(f"\n ~ | [{map_str}] try map unpack: {my_map.path}")
            try:
                map_registry[map_str] = extract_map_data(my_map.path, output_dir)
            except Exception as e:
                log_error(f" ~ | [{map_str}]: ERROR mpq_map_unpack ", e)

    output_lines = ["\n === Sorted Files by Map ==="]
    for map_name, mpq_entry in map_registry.items():
        output_lines.append(f"\n ~ | Map: {map_name}")
        if isinstance(mpq_entry, (MPQMapEntry, MPQReplayEntry)):
            for mime_type, files in mpq_entry.extracted_files.items():
                output_lines.append(f"\n ~ | [{map_name}]    MIME Type: {mime_type} (Total: {len(files)})")
                for file_info in files:
                    output_lines.append(f"\n ~ | [{map_name}]        - {file_info['file_name']} ({file_info['size_kb']} KB)")

    log_info("".join(output_lines))


def extract_all_replays_in_folder(replay_folder, output_dir):
    replay_folder = Path(replay_folder)
    output_dir = Path(output_dir)

    if not replay_folder.exists():
        print(f"Replay folder does not exist: {replay_folder}")
        return

    replay_files = list(replay_folder.glob("*.SC2Replay"))
    if not replay_files:
        print(f"No replay files found in: {replay_folder}")
        return

    print(f"Found {len(replay_files)} replays. Extracting...")

    i = 0
    for replay_file in replay_files:
        if i > 2: break # testing only, break after 2 replays
        print(f"Processing replay: {replay_file.name}")
        #replay_registry[replay_file.name] = extract_replay_data(replay_file, output_dir)
        replay_registry[replay_file.name] = extract_replay_data_w_sc2reader(str(replay_file), output_dir)
        i+=1

def extract_replays_main():
    replay_folder = f"{root_data_folder}replay_files"
    output_directory = f"{root_data_folder}extracted_replay_data"

    extract_all_replays_in_folder(replay_folder, output_directory)

    output_lines = ["\n === Sorted Files by Replay ==="]
    for replay_name, mpq_entry in replay_registry.items():
        output_lines.append(f"\n ~ | Replay: {replay_name}")
        if isinstance(mpq_entry, (MPQMapEntry, MPQReplayEntry)):
            for mime_type, files in mpq_entry.extracted_files.items():
                output_lines.append(f"\n ~ | [{replay_name}]    MIME Type: {mime_type} (Total: {len(files)})")
                for file_info in files:
                    output_lines.append(f"\n ~ | [{replay_name}]        - {file_info['file_name']} ({file_info['size_kb']} KB)")

    log_info("".join(output_lines))


# ================================

#extract_maps_main()
extract_replays_main()

