# region imports
# Third-party imports
import cv2
import numpy as np
# endregion imports


# -----------------------------------------------------------------------------------------------
# Motion detection
# -----------------------------------------------------------------------------------------------
class MotionDetector:
    ANALYSIS_WIDTH = 640
    # higher = background adapts faster (less sensitive to slow motion)
    BG_LEARNING_RATE = 0.02
    BLUR_KERNEL = (5, 5)
    # Pre-created 5×5 structuring element — equivalent to two iterations of the
    # default 3×3 kernel but computed only once instead of every frame.
    _DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def __init__(self, min_area: int, threshold: int):
        self.min_area = min_area
        self.threshold = threshold
        self._avg_frame = None
        # Geometry cache — populated on first frame; recomputed only if resolution changes.
        self._cached_w = None
        self._cached_h = None
        self._analysis_h = None
        self._sx = None
        self._sy = None
        self._scaled_min_area = None

    def detect(self, bgr_frame, orig_w: int, orig_h: int):
        """Detect motion in *bgr_frame* and return a list of (xmin,ymin,xmax,ymax) boxes.

        Accepts a BGR frame so the caller does not need a separate RGB→Gray conversion
        on top of the RGB→BGR conversion that is already required for drawing/recording.
        """
        # Cache scale factors — constant for a fixed resolution.
        if self._cached_w != orig_w or self._cached_h != orig_h:
            self._cached_w = orig_w
            self._cached_h = orig_h
            self._analysis_h = int(self.ANALYSIS_WIDTH * orig_h / orig_w)
            self._sx = orig_w / self.ANALYSIS_WIDTH
            self._sy = orig_h / self._analysis_h
            scale = (self.ANALYSIS_WIDTH * self._analysis_h) / (orig_w * orig_h)
            self._scaled_min_area = self.min_area * scale

        small = cv2.resize(bgr_frame, (self.ANALYSIS_WIDTH, self._analysis_h))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.BLUR_KERNEL, 0)

        if self._avg_frame is None:
            self._avg_frame = gray.astype(np.float32)
            return []

        cv2.accumulateWeighted(gray, self._avg_frame, self.BG_LEARNING_RATE)
        delta = cv2.absdiff(gray, cv2.convertScaleAbs(self._avg_frame))
        thresh = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        # Single dilation pass with the pre-built 5×5 kernel.
        thresh = cv2.dilate(thresh, self._DILATE_KERNEL, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        sx, sy = self._sx, self._sy
        scaled_min_area = self._scaled_min_area

        boxes = []
        for c in contours:
            if cv2.contourArea(c) < scaled_min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy)))
        return boxes
