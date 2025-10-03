"""
Gestures:
 - Thumbs Up     -> volume up
 - Thumbs Down   -> volume down
 - Open Palm     -> mute/unmute toggle
 - Fist (closed) -> play/pause toggle
"""
import time
import math
import collections
import cv2
import mediapipe as mp
import numpy as np
import ctypes


from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class SystemAudio:
    def __init__(self):
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))

    def get_volume_scalar(self) -> float:
        """0.0 - 1.0"""
        return self.volume.GetMasterVolumeLevelScalar()

    def set_volume_scalar(self, scalar: float):
        scalar = max(0.0, min(1.0, scalar))
        self.volume.SetMasterVolumeLevelScalar(scalar, None)

    def change_volume(self, delta: float):
        v = self.get_volume_scalar()
        self.set_volume_scalar(v + delta)

    def toggle_mute(self):
        current = self.volume.GetMute()
        self.volume.SetMute(not current, None)

    def is_muted(self) -> bool:
        return bool(self.volume.GetMute())

def send_play_pause():
    VK_MEDIA_PLAY_PAUSE = 0xB3
    # key down, key up
    ctypes.windll.user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 0, 0)
    time.sleep(0.02)
    ctypes.windll.user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 2, 0)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def landmarks_to_np(landmark_list, image_w, image_h):
    coords = np.zeros((21, 2), dtype=np.float32)
    for i, lm in enumerate(landmark_list.landmark):
        coords[i] = (lm.x * image_w, lm.y * image_h)
    return coords

def finger_is_up(coords, finger_tip_idx, finger_pip_idx):
    # coords are (x,y) with y increasing downward.
    # A finger is "up" if tip y is above (less) than pip y
    return coords[finger_tip_idx][1] < coords[finger_pip_idx][1]

def thumb_is_extended(coords, handedness_label):
    tip_x = coords[4][0]
    ip_x = coords[3][0]
    # if right hand, thumb extended to the right (tip_x > ip_x), else left
    if handedness_label == "Right":
        return tip_x > ip_x + 10  # small threshold in pixels
    else:
        return tip_x < ip_x - 10

def all_fingers_status(coords, handedness_label):
    # Returns tuple: (thumb_up_bool, index_bool, middle_bool, ring_bool, pinky_bool)
    idx = finger_is_up(coords, 8, 6)
    mid = finger_is_up(coords, 12, 10)
    ring = finger_is_up(coords, 16, 14)
    pinky = finger_is_up(coords, 20, 18)
    thumb = thumb_is_extended(coords, handedness_label)
    return (thumb, idx, mid, ring, pinky)

# Additional geometric helper: distance
def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# Gesture classification function:
def classify_gesture(coords, handedness_label):
    """
    Simple rule-based gesture detection:
      - Open palm: all 5 fingers extended
      - Fist: all 5 fingers folded
      - Thumbs up: thumb extended AND other 4 folded AND thumb tip above wrist
      - Thumbs down: similar but thumb tip below wrist
    Returns one of: "open_palm", "fist", "thumbs_up", "thumbs_down", None
    """
    thumb, idx, mid, ring, pinky = all_fingers_status(coords, handedness_label)
    stacked = [thumb, idx, mid, ring, pinky]
    # thresholds for robust detection:
    fingers_extended_count = sum(stacked)

    wrist = coords[0]
    thumb_tip = coords[4]

    # open palm
    if fingers_extended_count >= 4:
        return "open_palm"

    # fist (all folded)
    if fingers_extended_count == 0:
        return "fist"

    # thumbs up / down: thumb extended, others folded
    if thumb and not idx and not mid and not ring and not pinky:
        # check vertical relation between thumb tip and wrist
        if thumb_tip[1] < wrist[1] - 40:  # tip significantly above wrist => up
            return "thumbs_up"
        if thumb_tip[1] > wrist[1] + 40:  # significantly below => down
            return "thumbs_down"

    return None


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    audio = SystemAudio()
    print("Initial volume (0-1):", audio.get_volume_scalar(), "Muted:", audio.is_muted())

    # Debounce states: avoid repeated triggers continuously.
    last_action_time = {"thumbs_up": 0, "thumbs_down": 0, "open_palm": 0, "fist": 0}
    COOLDOWN = 1.0  # seconds between triggers for same gesture

    # Volume change step per thumbs gesture:
    VOLUME_STEP = 0.05

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror for natural interaction
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)
            gesture = None
            handedness_label = "Right"

            if results.multi_hand_landmarks:
                # use first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                if results.multi_handedness:
                    handedness_label = results.multi_handedness[0].classification[0].label

                coords = landmarks_to_np(hand_landmarks, w, h)
                gesture = classify_gesture(coords, handedness_label)

                # draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # annotate gesture on frame
                if gesture:
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

            # Act on gesture with cooldowns
            now = time.time()
            if gesture == "thumbs_up" and now - last_action_time["thumbs_up"] > COOLDOWN:
                print("[ACTION] Thumbs up -> volume up")
                audio.change_volume(VOLUME_STEP)
                last_action_time["thumbs_up"] = now

            elif gesture == "thumbs_down" and now - last_action_time["thumbs_down"] > COOLDOWN:
                print("[ACTION] Thumbs down -> volume down")
                audio.change_volume(-VOLUME_STEP)
                last_action_time["thumbs_down"] = now

            elif gesture == "open_palm" and now - last_action_time["open_palm"] > COOLDOWN:
                print("[ACTION] Open palm -> toggle mute")
                audio.toggle_mute()
                last_action_time["open_palm"] = now

            elif gesture == "fist" and now - last_action_time["fist"] > COOLDOWN:
                print("[ACTION] Fist -> play/pause")
                send_play_pause()
                last_action_time["fist"] = now

            # show volume on screen
            vol = audio.get_volume_scalar()
            muted = audio.is_muted()
            cv2.putText(frame, f"Volume: {int(vol*100)}% {'(MUTED)' if muted else ''}", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Gesture Controls", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

