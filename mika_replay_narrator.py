import os
import re
import sc2reader
from sc2reader.events import Event
from mew_utils.mew_log import log_info
from datetime import timedelta

from dataclasses import dataclass


REPLAY_FILE = "data/replay_files/3790730_Princess-Mika_muravevProtoss_ThunderbirdAIE.SC2Replay"

def format_replay(replay):
    return format_class_attributes(replay)

def load_replay(filepath):
    """Load the SC2Replay file with partial detail."""
    sc2reader.configure(debug=True)
    replay = sc2reader.load_replay(filepath, load_map=False, load_level=3)
    log_info(f"\n ~ | {format_replay(replay)}")
    return replay


@dataclass
class MewEvent:
    index: int
    time: float
    event: Event  # usually a sc2reader event

    def __str__(self):
        format_duration(self.event.seconds)
        time_str = f"{{{m}}}m{{{s}}}s"
        return f"[{self.index}] {time_str} ~ {str(self.event)}"


def extract_events(replay):
    """Extract basic game events from the replay."""
    events = []

    events.append((0, f"Game started on {replay.map_name}"))
    for player in replay.players:
        result = "Winner" if player.result == "Win" else "Loser"
        events.append((0, f"Player {player.name} ({player.pick_race}): {result}"))

    important_buildings = {
        "Hatchery", "Lair", "SpawningPool", "HydraliskDen", "UltraliskCavern",
        "Barracks", "Factory", "Starport", "CommandCenter"
    }

    for event in replay.events:
        log_info(f"\n ~ | [EVENT]: {event}")

        # if event.name == "UnitBornEvent" and hasattr(event, "unit"):
        #     name = event.unit.name.replace(" ", "")
        #     if name in important_buildings:
        #         seconds = event.second
        #         player = next((p.name for p in replay.players if p.pid == event.control_pid), "Unknown Player")
        #         events.append((seconds, f"{player} started building {event.unit.name}"))

    events.append((replay.game_length.seconds, "Game ended"))
    return events

def main():
    log_info("=== Mika Narrator ===")
    log_info(f"Loading replay: {REPLAY_FILE}")

    replay = load_replay(REPLAY_FILE)
    events = extract_events(replay)

    for time_sec, text in events:
        log_info(f"[{format_time(time_sec)}] {text}")

if __name__ == "__main__":
    main()
