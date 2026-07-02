# Air Canvas — Draw in the Air with Your Finger

A computer vision project that turns your webcam into a drawing pad. You
draw by moving your index finger in the air — no mouse, no touchscreen.

## How it works (step by step)

1. **Webcam captures video**, one frame (image) at a time, using OpenCV.
2. **MediaPipe** (a pretrained AI model from Google) looks at each frame and
   finds 21 key points on your hand — fingertips, knuckles, wrist. We don't
   train this model ourselves; it already knows what a hand looks like.
3. From those 21 points, simple math tells us **which fingers are up**:
   compare the fingertip's position to the joint just below it.
4. Based on your finger shape, the app decides what to do:
   - Only index finger up → **draw mode**: a line is drawn from your
     finger's previous position to its current position, frame after frame
     (that's how a smooth line appears, not a single dot).
   - Index + middle finger up → **selection mode**: move over the toolbar
     at the top to pick a color or the eraser.
   - Anything else (fist, all fingers up) → pen lifted, nothing drawn.
5. All the drawing is stored on a separate blank image called the
   **canvas**. Every frame, the canvas is merged on top of the live webcam
   feed so it looks like the drawing is "floating" in the air.

## Folder contents

```
air_canvas/
  hand_tracker.py     - wraps MediaPipe: finds hand points, detects fingers up
  air_canvas.py        - the main app (run this one)
  models/
    hand_landmarker.task  - pretrained hand-detection model (downloaded, not trained by us)
  requirements.txt     - list of libraries needed
```

## How to run it

From the `computer vision` folder, using the virtual environment already set up:

```powershell
.\venv\Scripts\Activate.ps1
cd air_canvas
python air_canvas.py
```

A window will open showing your webcam feed with a color toolbar on top.

## Controls

| Gesture / Key            | Action                          |
|---------------------------|----------------------------------|
| Index finger only up       | Draw                             |
| Index + middle finger up   | Selection mode (pick color/eraser by hovering over toolbar) |
| `c` key                    | Clear the whole canvas          |
| `s` key                    | Save your drawing as a PNG here |
| `q` key                    | Quit                             |

## Ideas to extend this later (optional)

- Add brush thickness control (e.g. pinky up = thicker line).
- Save drawings with a timestamped filename gallery.
- Add shape detection (recognize circles/squares you draw and redraw them perfectly).
- Swap MediaPipe's hand tracking for a custom-trained gesture classifier as a next step up in difficulty.
