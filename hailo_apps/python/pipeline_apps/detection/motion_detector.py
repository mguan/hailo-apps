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

    def __init__(self, min_area: int, threshold: int):
        self.min_area = min_area
        self.threshold = threshold
        self._avg_frame = None

    def detect(self, rgb_frame, orig_w: int, orig_h: int):
        analysis_h = int(self.ANALYSIS_WIDTH * orig_h / orig_w)
        small = cv2.resize(rgb_frame, (self.ANALYSIS_WIDTH, analysis_h))
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, self.BLUR_KERNEL, 0)

        if self._avg_frame is None:
            self._avg_frame = gray.astype(np.float32)
            return []

        cv2.accumulateWeighted(gray, self._avg_frame, self.BG_LEARNING_RATE)
        delta = cv2.absdiff(gray, cv2.convertScaleAbs(self._avg_frame))
        thresh = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scale = (self.ANALYSIS_WIDTH * analysis_h) / (orig_w * orig_h)
        scaled_min_area = self.min_area * scale
        sx = orig_w / self.ANALYSIS_WIDTH
        sy = orig_h / analysis_h

        boxes = []
        for c in contours:
            if cv2.contourArea(c) < scaled_min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy)))
        return boxes
