import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import math
import os

# === Load all puzzle pieces ===
folder_path = r'C:\Users\gharib\OneDrive - Alexandria University\Desktop\Medo\Machine Learning\Computer Vision\Advanced Computer Vision\Puzzle Pieces'
image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])[:4]
puzzle_pieces = [cv.imread(os.path.join(folder_path, file), cv.IMREAD_UNCHANGED) for file in image_files]

# === Camera Setup ===
capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 1080)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
previous_time = 0

# === Initial puzzle positions: one per corner ===
positions = [
    [50, 50],        # top-left
    [810, 50],       # top-right
    [785, 375],      # bottom-right
    [50, 350]        # bottom-left
]
dragging_index = None
drag_offset = (0, 0)
snap_threshold = 30

def putImage(frame, img, position):
    frame_h, frame_w = frame.shape[:2]
    img_h, img_w = img.shape[:2]
    x, y = position

    # Frame coordinates (clip to frame size)
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + img_w, frame_w)
    y2 = min(y + img_h, frame_h)

    # Image coordinates (adjust for clipping)
    overlay_x1 = max(0, -x)
    overlay_y1 = max(0, -y)
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    # Skip if the overlay area is invalid
    if x1 >= x2 or y1 >= y2 or overlay_x1 >= overlay_x2 or overlay_y1 >= overlay_y2:
        return frame

    roi = frame[y1:y2, x1:x2]
    overlay_region = img[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    if overlay_region.shape[2] == 4:
        alpha = overlay_region[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (
                roi[:, :, c] * (1 - alpha) + overlay_region[:, :, c] * alpha
            ).astype(np.uint8)
    else:
        roi[:] = overlay_region

    frame[y1:y2, x1:x2] = roi
    return frame

def isHolding(x1, y1, x2, y2, threshold=40):
    distance = math.hypot(x2 - x1, y2 - y1)
    return distance < threshold

def point_inside(pos, top_left, size):
    x, y = pos
    tx, ty = top_left
    w, h = size
    return tx <= x <= tx + w and ty <= y <= ty + h

def snap_to_other_pieces(index, positions, pieces, threshold=30, overlap=25):
    current_piece = pieces[index]
    current_pos = positions[index]
    current_w, current_h = current_piece.shape[1], current_piece.shape[0]

    for i, (other_piece, other_pos) in enumerate(zip(pieces, positions)):
        if i == index:
            continue
        other_w, other_h = other_piece.shape[1], other_piece.shape[0]

        # Current piece edges
        cx1, cy1 = current_pos
        cx2, cy2 = cx1 + current_w, cy1 + current_h

        # Other piece edges
        ox1, oy1 = other_pos
        ox2, oy2 = ox1 + other_w, oy1 + other_h

        # Snap left to right
        if abs(cx1 - ox2) < threshold and abs(cy1 - oy1) < threshold:
            positions[index] = [ox2 - overlap, oy1]
            return
        # Snap right to left
        if abs(cx2 - ox1) < threshold and abs(cy1 - oy1) < threshold:
            positions[index] = [ox1 - current_w + overlap, oy1]
            return
        # Snap top to bottom
        if abs(cy1 - oy2) < threshold and abs(cx1 - ox1) < threshold:
            positions[index] = [ox1, oy2 - overlap]
            return
        # Snap bottom to top
        if abs(cy2 - oy1) < threshold and abs(cx1 - ox1) < threshold:
            positions[index] = [ox1, oy1 - current_h + overlap]
            return
    current_piece = pieces[index]
    current_pos = positions[index]
    current_w, current_h = current_piece.shape[1], current_piece.shape[0]

    for i, (other_piece, other_pos) in enumerate(zip(pieces, positions)):
        if i == index:
            continue
        other_w, other_h = other_piece.shape[1], other_piece.shape[0]

        # Current piece edges
        cx1, cy1 = current_pos
        cx2, cy2 = cx1 + current_w, cy1 + current_h

        # Other piece edges
        ox1, oy1 = other_pos
        ox2, oy2 = ox1 + other_w, oy1 + other_h

        # Snap left to right
        if abs(cx1 - ox2) < threshold and abs(cy1 - oy1) < threshold:
            positions[index] = [ox2, oy1]
            return
        # Snap right to left
        if abs(cx2 - ox1) < threshold and abs(cy1 - oy1) < threshold:
            positions[index] = [ox1 - current_w, oy1]
            return
        # Snap top to bottom
        if abs(cy1 - oy2) < threshold and abs(cx1 - ox1) < threshold:
            positions[index] = [ox1, oy2]
            return
        # Snap bottom to top
        if abs(cy2 - oy1) < threshold and abs(cx1 - ox1) < threshold:
            positions[index] = [ox1, oy1 - current_h]
            return

# === Main Loop ===
if not capture.isOpened():
    print('Error: Could not open the camera')
else:
    while True:
        success, frame = capture.read()
        if not success:
            break

        frame = cv.flip(frame, 1)
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frameRGB)

        h, w, _ = frame.shape
        index_pos = None
        thumb_pos = None

        if results.multi_hand_landmarks:
            for handlms in results.multi_hand_landmarks:
                thumb_tip = handlms.landmark[4]
                index_tip = handlms.landmark[8]

                thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_pos = (int(index_tip.x * w), int(index_tip.y * h))

                cv.circle(frame, thumb_pos, 10, (255, 0, 255), -1)
                cv.circle(frame, index_pos, 10, (0, 255, 0), -1)

                mp_draw.draw_landmarks(frame, handlms, mp_hands.HAND_CONNECTIONS)

        if index_pos and thumb_pos:
            if isHolding(*thumb_pos, *index_pos):
                cv.putText(frame, 'Holding', (500, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                if dragging_index is None:
                    for i, (piece, pos) in enumerate(zip(puzzle_pieces, positions)):
                        if point_inside(index_pos, pos, piece.shape[:2][::-1]):
                            dragging_index = i
                            drag_offset = (index_pos[0] - pos[0], index_pos[1] - pos[1])
                            break
            else:
                dragging_index = None
                cv.putText(frame, 'Not Holding', (500, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        else:
            dragging_index = None

        if dragging_index is not None and index_pos:
            positions[dragging_index][0] = index_pos[0] - drag_offset[0]
            positions[dragging_index][1] = index_pos[1] - drag_offset[1]

            # Try snapping to neighbors
            snap_to_other_pieces(dragging_index, positions, puzzle_pieces, snap_threshold)

        for piece, pos in zip(puzzle_pieces, positions):
            frame = putImage(frame, piece, tuple(pos))

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv.putText(frame, f'FPS: {int(fps)}', (10, 30), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv.imshow("Live Cam", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

capture.release()
cv.destroyAllWindows()
