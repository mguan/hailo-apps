import time

from hailo_apps.python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)

# COCO pose keypoint indices
KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Minimum seconds between fall alert logs for a single tracked person
INITIAL_FALL_DEBOUNCE_SECONDS = 5.0
MAX_FALL_DEBOUNCE_SECONDS = 3600.0

# Seconds a person must remain untracked as falling to reset the alarm backoff
FALL_RESET_SECONDS = 5.0


class FallDetector:
    def __init__(self, safe_zones=None):
        self.safe_zones = safe_zones or []
        self.last_alert_time = {}
        self.last_seen_fallen_time = {}
        self.fall_backoff = {}

    def is_fall_detected(self, track_id, bbox, points):
        # 1. Bounding box wider than tall -> person is horizontal
        is_horizontal_shape = bbox.width() / (bbox.height() + 1e-6) > 1.0

        # 2. Torso angle: horizontal when hip-to-shoulder dx exceeds dy
        left_shoulder = points[KEYPOINTS["left_shoulder"]]
        right_shoulder = points[KEYPOINTS["right_shoulder"]]
        left_hip = points[KEYPOINTS["left_hip"]]
        right_hip = points[KEYPOINTS["right_hip"]]

        shoulder_mid_x = (left_shoulder.x() + right_shoulder.x()) / 2
        shoulder_mid_y = (left_shoulder.y() + right_shoulder.y()) / 2
        hip_mid_x = (left_hip.x() + right_hip.x()) / 2
        hip_mid_y = (left_hip.y() + right_hip.y()) / 2

        is_torso_horizontal = abs(hip_mid_x - shoulder_mid_x) > abs(hip_mid_y - shoulder_mid_y)

        if not (is_horizontal_shape and is_torso_horizontal):
            return False

        # Check whether the person is inside a predefined safe zone (e.g. a bed)
        mid_x = bbox.xmin() + bbox.width() / 2.0
        mid_y = bbox.ymin() + bbox.height() / 2.0

        for z_xmin, z_ymin, z_xmax, z_ymax in self.safe_zones:
            if z_xmin <= mid_x <= z_xmax and z_ymin <= mid_y <= z_ymax:
                hailo_logger.debug("Horizontal pose in safe zone (Person ID %d) — resting.", track_id)
                return False

        return True

    def is_fall_resolved(self, track_id) -> bool:
        """Returns True once when a previously-falling person has not been seen falling for FALL_RESET_SECONDS."""
        if track_id not in self.last_seen_fallen_time:
            return False
        if time.time() - self.last_seen_fallen_time[track_id] > FALL_RESET_SECONDS:
            del self.last_seen_fallen_time[track_id]
            self.fall_backoff.pop(track_id, None)
            return True
        return False

    def check_alert_throttle(self, track_id) -> bool:
        """
        Updates the internal state tracking the time a fall was detected for a person,
        and applies an exponential backoff delay to determine if a new alert should be sent.
        
        Returns:
            bool: True if an alert should be dispatched now, False if the alert is throttled.
        """
        current_time = time.time()

        # If the person has not been seen falling for over FALL_RESET_SECONDS, reset their backoff
        if current_time - self.last_seen_fallen_time.get(track_id, 0.0) > FALL_RESET_SECONDS:
            self.fall_backoff[track_id] = INITIAL_FALL_DEBOUNCE_SECONDS

        self.last_seen_fallen_time[track_id] = current_time

        # Exponential backoff: alert at most once every current_backoff seconds per tracked person
        current_backoff = self.fall_backoff.get(track_id, INITIAL_FALL_DEBOUNCE_SECONDS)

        if current_time - self.last_alert_time.get(track_id, 0.0) > current_backoff:
            self.last_alert_time[track_id] = current_time
            self.fall_backoff[track_id] = min(current_backoff * 2, MAX_FALL_DEBOUNCE_SECONDS)
            return True

        return False
