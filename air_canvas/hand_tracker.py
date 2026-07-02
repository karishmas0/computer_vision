"""
hand_tracker.py
----------------
A small wrapper around MediaPipe's HandLandmarker so the main app
doesn't need to deal with MediaPipe's API directly.

What it gives you, per video frame:
  - the (x, y) pixel position of all 21 hand landmarks (skeleton points)
  - which fingers are currently held up (as a list of 0/1)

Landmark numbering (MediaPipe standard), for reference:
  0  = wrist
  4  = thumb tip        8  = index fingertip
  12 = middle fingertip 16 = ring fingertip     20 = pinky fingertip
"""

import os
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")

# Landmark indices for the tip and the knuckle-below-tip of each finger.
# We compare tip vs. the joint two below it to decide "is this finger up?"
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]  # the joint just below each tip


class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.6, tracking_confidence=0.5):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0
        self.landmark_points = []  # list of (x, y) in pixels, empty if no hand found

    def process(self, frame_bgr):
        """Run detection on one BGR frame (as read by OpenCV). Must be called once per frame, in order."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._timestamp_ms += 33  # ~30fps step; the API just needs increasing numbers
        result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)

        self.landmark_points = []
        if result.hand_landmarks:
            h, w = frame_bgr.shape[:2]
            hand = result.hand_landmarks[0]  # only using the first detected hand
            self.landmark_points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

        return self.landmark_points

    def fingers_up(self):
        """Returns [thumb, index, middle, ring, pinky] as 1 (up) or 0 (down)."""
        if not self.landmark_points:
            return [0, 0, 0, 0, 0]

        fingers = []

        # Thumb: compare x, not y, since it folds sideways rather than up/down.
        thumb_tip_x = self.landmark_points[4][0]
        thumb_pip_x = self.landmark_points[3][0]
        wrist_x = self.landmark_points[0][0]
        if wrist_x < thumb_pip_x:  # hand is roughly facing camera, thumb on the right
            fingers.append(1 if thumb_tip_x > thumb_pip_x else 0)
        else:
            fingers.append(1 if thumb_tip_x < thumb_pip_x else 0)

        # Other four fingers: up if the tip is higher (smaller y) than the joint below it.
        for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
            fingers.append(1 if self.landmark_points[tip][1] < self.landmark_points[pip][1] else 0)

        return fingers

    def close(self):
        self.landmarker.close()
