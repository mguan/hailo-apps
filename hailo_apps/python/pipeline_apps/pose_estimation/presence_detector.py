import time
from hailo_apps.python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)

# Minimum seconds a person must be detected to trigger an entry event
ENTRY_DEBOUNCE_SECONDS = 0.5
# Minimum seconds a person must be missing to trigger an exit event
EXIT_DEBOUNCE_SECONDS = 2.0

class PresenceDetector:
    def __init__(self, door_zones=None):
        """
        door_zones: List of (xmin, ymin, xmax, ymax) defining regions
                    where presence triggers entry/exit. If None or empty,
                    the entire frame is monitored. (values from 0.0 to 1.0)
        """
        self.door_zones = door_zones or []
        self.first_seen_time = {}
        self.last_seen_time = {}
        self.active_presences = set()

    def update(self, current_tracks):
        """
        current_tracks: dict mapping track_id -> bbox
        Returns a list of event dictionaries: [{"type": "entry"/"exit", "track_id": id}]
        """
        current_time = time.time()
        events = []
        active_tracks_in_zone = {}

        for track_id, bbox in current_tracks.items():
            # Check if inside a door zone
            in_zone = False
            if self.door_zones:
                # Calculate center point in relative coordinates 0.0 to 1.0
                mid_x = bbox.xmin() + bbox.width() / 2.0
                mid_y = bbox.ymin() + bbox.height() / 2.0
                for z_xmin, z_ymin, z_xmax, z_ymax in self.door_zones:
                    if z_xmin <= mid_x <= z_xmax and z_ymin <= mid_y <= z_ymax:
                        in_zone = True
                        break
            else:
                in_zone = True

            if in_zone:
                active_tracks_in_zone[track_id] = bbox

        # Update tracking for active detections
        for track_id, bbox in active_tracks_in_zone.items():
            if track_id not in self.first_seen_time:
                self.first_seen_time[track_id] = current_time
            self.last_seen_time[track_id] = current_time

            # Check for entry event
            if track_id not in self.active_presences:
                if current_time - self.first_seen_time[track_id] >= ENTRY_DEBOUNCE_SECONDS:
                    self.active_presences.add(track_id)
                    events.append({"type": "entry", "track_id": track_id})

        # Check for exit events
        exited_tracks = []
        for track_id in list(self.first_seen_time.keys()):
            if track_id not in active_tracks_in_zone:
                if current_time - self.last_seen_time.get(track_id, current_time) > EXIT_DEBOUNCE_SECONDS:
                    if track_id in self.active_presences:
                        events.append({"type": "exit", "track_id": track_id})
                        self.active_presences.remove(track_id)
                    exited_tracks.append(track_id)

        for track_id in exited_tracks:
            del self.first_seen_time[track_id]
            del self.last_seen_time[track_id]

        return events
