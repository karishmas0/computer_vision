"""
air_canvas.py
-------------
Draw in the air using your index finger, tracked live through your webcam.

Controls (done with your hand, no keyboard/mouse):
  - Hold up ONLY your index finger              -> draw
  - Hold up your index + middle finger together  -> "selection mode":
        move over the top toolbar to pick a color or the eraser
  - Any other hand shape (fist, all fingers up)  -> pen lifted, nothing drawn

Keyboard shortcuts:
  c -> clear the canvas
  s -> save your drawing as a PNG (in this folder)
  q -> quit
"""

import os
import time
import cv2
import numpy as np

from hand_tracker import HandTracker

CAM_WIDTH, CAM_HEIGHT = 1280, 720
TOOLBAR_HEIGHT = 100
DRAW_THICKNESS = 8
ERASER_THICKNESS = 50

# name, BGR color, x-range on the toolbar (filled in below)
COLORS = [
    ("Blue",   (255, 0, 0)),
    ("Green",  (0, 255, 0)),
    ("Red",    (0, 0, 255)),
    ("Yellow", (0, 255, 255)),
    ("Eraser", (0, 0, 0)),
]


def build_toolbar(width):
    """Draws the row of color swatches shown at the top of the window."""
    toolbar = np.zeros((TOOLBAR_HEIGHT, width, 3), dtype=np.uint8)
    toolbar[:] = (40, 40, 40)  # dark grey background

    box_width = width // len(COLORS)
    boxes = []  # (x_start, x_end, name, color)
    for i, (name, color) in enumerate(COLORS):
        x_start = i * box_width
        x_end = x_start + box_width
        swatch_color = color if name != "Eraser" else (200, 200, 200)
        cv2.rectangle(toolbar, (x_start + 10, 10), (x_end - 10, TOOLBAR_HEIGHT - 10), swatch_color, -1)
        cv2.putText(toolbar, name, (x_start + 15, TOOLBAR_HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255) if name != "Eraser" else (0, 0, 0), 2)
        boxes.append((x_start, x_end, name, color))
    return toolbar, boxes


def color_at(x, boxes):
    for x_start, x_end, name, color in boxes:
        if x_start <= x < x_end:
            return name, color
    return None, None


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print("Could not open webcam. Is it being used by another app?")
        return

    tracker = HandTracker(max_hands=1)
    toolbar, boxes = None, None
    canvas = None

    draw_color = COLORS[2][1]  # start with Red
    draw_name = COLORS[2][0]
    prev_point = None

    print("Air Canvas running. Press 'q' in the window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror, feels natural
        h, w = frame.shape[:2]

        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            toolbar, boxes = build_toolbar(w)

        landmarks = tracker.process(frame)

        if landmarks:
            fingers = tracker.fingers_up()
            index_up = fingers[1] == 1
            middle_up = fingers[2] == 1
            index_tip = landmarks[8]

            if index_up and middle_up:
                # Selection mode: hovering over the toolbar picks a color.
                prev_point = None
                cv2.circle(frame, index_tip, 12, draw_color, cv2.FILLED)
                if index_tip[1] < TOOLBAR_HEIGHT:
                    name, color = color_at(index_tip[0], boxes)
                    if name:
                        draw_name, draw_color = name, color

            elif index_up and not middle_up:
                # Draw mode.
                if index_tip[1] > TOOLBAR_HEIGHT:  # don't draw over the toolbar
                    thickness = ERASER_THICKNESS if draw_name == "Eraser" else DRAW_THICKNESS
                    if prev_point is None:
                        prev_point = index_tip
                    cv2.line(canvas, prev_point, index_tip, draw_color, thickness)
                    prev_point = index_tip
                else:
                    prev_point = None

            else:
                prev_point = None
        else:
            prev_point = None

        # Merge the canvas onto the live frame: wherever the canvas has ink,
        # show that instead of the webcam pixel; everywhere else, show the webcam as-is.
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask_inv = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, mask_inv)
        frame = cv2.bitwise_or(frame, canvas)

        frame[0:TOOLBAR_HEIGHT, 0:w] = toolbar
        cv2.putText(frame, f"Selected: {draw_name}", (10, TOOLBAR_HEIGHT + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Air Canvas - press q to quit, c to clear, s to save", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas[:] = 0
        elif key == ord('s'):
            filename = f"drawing_{int(time.time())}.png"
            path = os.path.join(os.path.dirname(__file__), filename)
            cv2.imwrite(path, canvas)
            print(f"Saved: {path}")

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
