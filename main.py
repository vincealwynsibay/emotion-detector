"""
Combined Emotion Detection: Keras + MediaPipe

Detections (priority order):
  1. 67          — two palms facing up 🫴🫴 (MediaPipe Hands)
  2. shaka       — thumb + pinky out, middle three folded 🤙 (MediaPipe Hands)
  3. yawn        — mouth very wide open
  4. tongue      — mouth slightly open
  5. left wink   — left eye closed, right open
  6. right wink  — right eye closed, left open
  7. one brow up — only ONE eyebrow raised (avoids intersecting with surprise)
  8. Keras CNN   — anger, disgust, fear, happiness, sadness, surprise, neutral
"""

import cv2
import numpy as np
import random
import time
from pathlib import Path

from keras.models import load_model
import mediapipe as mp
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

ASSETS_DIR = Path("assets")
MODEL_PATH  = "model.keras"

model = load_model(MODEL_PATH)
label_to_text = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "sadness",
    5: "surprise",
    6: "neutral",
}

hands_detector = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,       
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

MOUTH_OPEN_THRESHOLD  = 0.03
YAWN_THRESHOLD        = 0.07
WINK_RATIO_THRESHOLD  = 2.5
WINK_MIN_OPEN         = 0.015
ONE_BROW_THRESHOLD    = 0.22
BROW_ASYMM_THRESHOLD  = 0.02

def _ear(top_lm, bot_lm):
    return abs(top_lm.y - bot_lm.y)


def detect_yawn(lm):
    return abs(lm.landmark[13].y - lm.landmark[14].y) > YAWN_THRESHOLD


def detect_tongue(lm):
    return abs(lm.landmark[13].y - lm.landmark[14].y) > MOUTH_OPEN_THRESHOLD


def detect_wink(lm):
    l = _ear(lm.landmark[159], lm.landmark[145])
    r = _ear(lm.landmark[386], lm.landmark[374])
    if l < 1e-4 or r < 1e-4:
        return False, None
    ratio = l / r if l > r else r / l
    if ratio > WINK_RATIO_THRESHOLD:
        if l > r and l > WINK_MIN_OPEN:
            return True, "right wink"
        if r > l and r > WINK_MIN_OPEN:
            return True, "left wink"
    return False, None


def detect_one_brow_raise(lm):
    face_h = abs(lm.landmark[10].y - lm.landmark[152].y)
    if face_h < 1e-4:
        return False, None
    l_dist = (lm.landmark[159].y - lm.landmark[105].y) / face_h
    r_dist = (lm.landmark[386].y - lm.landmark[334].y) / face_h
    asymmetry = abs(l_dist - r_dist)
    if asymmetry < BROW_ASYMM_THRESHOLD:
        return False, None
    if l_dist > ONE_BROW_THRESHOLD and l_dist > r_dist:
        return True, "left brow up"
    if r_dist > ONE_BROW_THRESHOLD and r_dist > l_dist:
        return True, "right brow up"
    return False, None


def _finger_extended(lm, tip, pip):
    """Returns True if a finger is extended (tip above pip in image coords)."""
    return lm[tip].y < lm[pip].y


def _finger_folded(lm, tip, pip):
    return lm[tip].y > lm[pip].y


def _is_palm_up(hand_lm):
    """Palm faces up: wrist below mid-MCP and palm normal points toward camera."""
    lm      = hand_lm.landmark
    wrist   = lm[0]
    mid_mcp = lm[9]

    # Wrist must be lower than mid knuckle (hand held up)
    if wrist.y <= mid_mcp.y:
        return False

    # All fingers roughly extended (open palm)
    fingers_open = all([
        _finger_extended(lm, 8, 6),
        _finger_extended(lm, 12, 10),
        _finger_extended(lm, 16, 14),
        _finger_extended(lm, 20, 18),
    ])

    if not fingers_open:
        return False

    # Palm up: middle fingertip z should be less than wrist z
    # (fingertips closer to camera than wrist when palm faces up)
    return lm[12].z < wrist.z


def detect_67(hand_results):
    """Both hands with palms facing up 🫴🫴"""
    if not hand_results or not hand_results.multi_hand_landmarks:
        return False
    if len(hand_results.multi_hand_landmarks) < 2:
        return False
    return all(_is_palm_up(h) for h in hand_results.multi_hand_landmarks)


def _is_shaka(hand_lm):
    """
    Shaka 🤙: thumb + pinky extended, index + middle + ring folded.
    MediaPipe hand landmarks:
      Thumb tip=4, ip=3
      Index tip=8, pip=6
      Middle tip=12, pip=10
      Ring tip=16, pip=14
      Pinky tip=20, pip=18
    """
    lm = hand_lm.landmark

    # Thumb: tip should be far from ring finger tip (works front AND back)
    thumb_extended = (
        abs(lm[4].x - lm[5].x) > 0.08 or  # front of hand
        abs(lm[4].x - lm[17].x) > 0.08     # back of hand
    )

    pinky_extended = _finger_extended(lm, 20, 18)
    index_folded   = _finger_folded(lm, 8, 6)
    middle_folded  = _finger_folded(lm, 12, 10)
    ring_folded    = _finger_folded(lm, 16, 14)

    return thumb_extended and pinky_extended and index_folded and middle_folded and ring_folded



def detect_shaka(hand_results):
    """Returns True if any detected hand shows the shaka sign 🤙"""
    if not hand_results or not hand_results.multi_hand_landmarks:
        return False
    return any(_is_shaka(h) for h in hand_results.multi_hand_landmarks)

def detect_one_brow_raise(lm):
    face_h = abs(lm.landmark[10].y - lm.landmark[152].y)
    if face_h < 1e-4:
        return False, None

    # Distance from eye center to brow
    l_dist = (lm.landmark[145].y - lm.landmark[105].y) / face_h
    r_dist = (lm.landmark[374].y - lm.landmark[334].y) / face_h

    asymmetry = abs(l_dist - r_dist)
    if asymmetry < 0.02:
        return False, None

    if l_dist > r_dist and l_dist > 0.12:
        return True, "left brow up"
    if r_dist > l_dist and r_dist > 0.12:
        return True, "right brow up"
    return False, None

def _is_one_finger(hand_lm):
    """Only index finger extended, rest folded."""
    lm = hand_lm.landmark
    index_up  = _finger_extended(lm, 8, 6)
    middle_dn = _finger_folded(lm, 12, 10)
    ring_dn   = _finger_folded(lm, 16, 14)
    pinky_dn  = _finger_folded(lm, 20, 18)
    return index_up and middle_dn and ring_dn and pinky_dn


def _is_thumbs_up(hand_lm):
    """Thumb extended upward, all fingers folded."""
    lm = hand_lm.landmark
    # Thumb tip above wrist
    thumb_up  = lm[4].y < lm[2].y
    # Thumb tip clearly above index MCP
    thumb_high = lm[4].y < lm[5].y
    index_dn  = _finger_folded(lm, 8, 6)
    middle_dn = _finger_folded(lm, 12, 10)
    ring_dn   = _finger_folded(lm, 16, 14)
    pinky_dn  = _finger_folded(lm, 20, 18)
    return thumb_up and thumb_high and index_dn and middle_dn and ring_dn and pinky_dn


def detect_one_finger(hand_results):
    if not hand_results or not hand_results.multi_hand_landmarks:
        return False
    return any(_is_one_finger(h) for h in hand_results.multi_hand_landmarks)


def detect_thumbs_up(hand_results):
    if not hand_results or not hand_results.multi_hand_landmarks:
        return False
    return any(_is_thumbs_up(h) for h in hand_results.multi_hand_landmarks)

def _is_one_finger(hand_lm):
    """
    One finger up ☝️:
    Index extended, all other fingers folded.
    """
    lm = hand_lm.landmark

    index_extended = _finger_extended(lm, 8, 6)

    middle_folded = _finger_folded(lm, 12, 10)
    ring_folded   = _finger_folded(lm, 16, 14)
    pinky_folded  = _finger_folded(lm, 20, 18)

    # Thumb folded toward palm
    thumb_folded = abs(lm[4].x - lm[3].x) < 0.04

    return (
        index_extended and
        middle_folded and
        ring_folded and
        pinky_folded and
        thumb_folded
    )

def detect_one_finger(hand_results):
    """Returns True if any hand shows ☝️ gesture"""
    if not hand_results or not hand_results.multi_hand_landmarks:
        return False

    return any(_is_one_finger(h) for h in hand_results.multi_hand_landmarks)

DISPLAY_SIZE = (400, 400)
images: dict = {}


def load_image(path, size=DISPLAY_SIZE):
    img = cv2.imread(str(path))
    if img is not None:
        return cv2.resize(img, size)
    return None


def find_image(name, ext=".png"):
    for base in [ASSETS_DIR, Path(".")]:
        p = base / f"{name}{ext}"
        if p.exists():
            return load_image(p)
    return None


for name in label_to_text.values():
    img = find_image(name, ".png")
    if img is not None:
        images[name] = img

extra_names = [
    "cat-shock", "cat-tongue", "cat-glare", "larry",
    "cat-yawn", "cat-wink", "cat-brow", "palms-up", "shaka",
    "one-finger", "thumbs-up",  
]
for name in extra_names:
    for ext in [".jpeg", ".jpg", ".png"]:
        img = find_image(name, ext)
        if img is None:
            img = find_image(name.replace("-", "_"), ext)
        if img is not None:
            images[name] = img
            break

none_img = images.get("neutral")
if none_img is None:
    none_img = images.get("larry")
if none_img is None and images:
    none_img = next(iter(images.values()))

IMAGE_POOLS = {
    "67":            ["palms-up"],
    "shaka":         ["shaka"],
    "yawn":          ["cat-yawn"],
    "tongue":        ["cat-tongue"],
    "left wink":     ["cat-wink"],
    "right wink":    ["cat-wink"],
    "left brow up":  ["cat-brow"],
    "right brow up": ["cat-brow"],
    "one finger": ["one-finger"],
    "thumbs up":  ["thumbs-up"],
    "anger":         ["anger"],
    "disgust":       ["disgust"],
    "fear":          ["fear"],
    "happiness":     ["happiness"],
    "sadness":       ["sadness"],
    "surprise":      ["surprise"],
    "neutral":       ["neutral"],
}

HOLD_FRAMES        = 10
_last_stable_state = None
_current_img       = None
_candidate_state   = None
_candidate_count   = 0


def pick_image(state: str):
    global _last_stable_state, _current_img, _candidate_state, _candidate_count

    if state == _last_stable_state:
        _candidate_state = None
        _candidate_count = 0
        return _current_img if _current_img is not None else none_img

    if state == _candidate_state:
        _candidate_count += 1
    else:
        _candidate_state = state
        _candidate_count = 1

    if _candidate_count >= HOLD_FRAMES:
        pool  = IMAGE_POOLS.get(state, ["neutral", "larry"])
        valid = [k for k in pool if k in images]
        _current_img       = images[random.choice(valid)] if valid else none_img
        _last_stable_state = state
        _candidate_state   = None
        _candidate_count   = 0

    return _current_img if _current_img is not None else none_img


def main():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if none_img is None:
        print("Warning: No images loaded. Add assets to the assets/ folder.")

    KERAS_SKIP         = 6
    frame_count        = 0
    last_keras_emotion = "neutral"
    last_keras_conf    = 0.0
    fps_time           = time.time()
    fps                = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.flip(frame, 1)
        process_frame = cv2.resize(frame, (320, 240)) 
        gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
        rgb  = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        emotion_img = none_img
        state_label = "no face"
        conf_label  = ""

        rgb.flags.writeable = False
        hand_results = hands_detector.process(rgb)

        face_result    = face_mesh.process(rgb)
        rgb.flags.writeable = True
        face_landmarks = face_result.multi_face_landmarks

        run_keras = (frame_count % KERAS_SKIP == 0)
        faces = []
        if run_keras:
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5
            )

        mp_matched = False

        if detect_67(hand_results):
            state_label = "67"
            emotion_img = pick_image("67")
            mp_matched  = True

        elif detect_shaka(hand_results):
            state_label = "shaka"
            emotion_img = pick_image("shaka")
            mp_matched  = True

        elif detect_one_finger(hand_results):
            state_label = "one finger"
            emotion_img = pick_image("one finger")
            mp_matched  = True
        elif detect_thumbs_up(hand_results):
            state_label = "thumbs up"
            emotion_img = pick_image("thumbs up")
            mp_matched  = True

        elif detect_one_finger(hand_results):
            state_label = "one finger"
            emotion_img = pick_image("one finger")
            mp_matched  = True
        elif face_landmarks:
            lm = face_landmarks[0]

            if detect_yawn(lm):
                state_label = "yawn"
                mp_matched  = True

            elif detect_tongue(lm):
                state_label = "tongue"
                mp_matched  = True

            else:
                wink_ok, wink_side = detect_wink(lm)
                if wink_ok:
                    state_label = wink_side
                    mp_matched  = True
                else:
                    brow_ok, brow_side = detect_one_brow_raise(lm)
                    if brow_ok:
                        state_label = brow_side
                        mp_matched  = True

            if mp_matched:
                emotion_img = pick_image(state_label)

            elif run_keras and len(faces) > 0:
                faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, fw, fh = faces_sorted[0]
                roi = gray[y:y + fh, x:x + fw]
                roi = cv2.resize(roi, (48, 48)).astype("float32") / 255.0
                roi = np.expand_dims(roi, axis=(0, -1))
                preds = model.predict(roi, verbose=0)[0]
                idx = int(np.argmax(preds))
                last_keras_emotion = label_to_text[idx]
                last_keras_conf    = float(preds[idx])
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

            if not mp_matched:
                state_label = last_keras_emotion
                conf_label  = f"{last_keras_conf * 100:.0f}%"
                emotion_img = pick_image(state_label)

        elif run_keras and len(faces) > 0:
            faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, fw, fh = faces_sorted[0]
            roi = gray[y:y + fh, x:x + fw]
            roi = cv2.resize(roi, (48, 48)).astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=(0, -1))
            preds = model.predict(roi, verbose=0)[0]
            idx = int(np.argmax(preds))
            last_keras_emotion = label_to_text[idx]
            last_keras_conf    = float(preds[idx])
            state_label = last_keras_emotion
            conf_label  = f"{last_keras_conf * 100:.0f}%"
            emotion_img = pick_image(state_label)
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

        if emotion_img is None:
            emotion_img = np.zeros((*DISPLAY_SIZE, 3), dtype=np.uint8)
            cv2.putText(emotion_img, "Add assets", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- FPS ---
        now      = time.time()
        fps      = 0.9 * fps + 0.1 * (1.0 / max(now - fps_time, 1e-4))
        fps_time = now

        # --- Compose display ---
        frame_resized = cv2.resize(frame, DISPLAY_SIZE)
        combined = np.hstack((frame_resized, emotion_img))
        cv2.rectangle(combined, (0, 0), (combined.shape[1], 35), (0, 0, 0), -1)
        cv2.putText(combined, state_label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        if conf_label:
            cv2.putText(combined, conf_label, (200, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        cv2.putText(combined, f"FPS {fps:.1f}", (470, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.imshow("Emotion Detection", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()