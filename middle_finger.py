import time
import cv2
import numpy as np
import mediapipe as mp
import ctypes

show_window = False

locking_enabled = True
last_lock_time = 0
debounce_time = 3

main_color = (85, 185, 247)
main_alpha = 0.3
main_pad = 20
main_radius = 5

mp_hands = mp.solutions.hands # type: ignore
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils # type: ignore

def draw_text(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=2, pad=6):
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    
    rect_tl = (x - int(pad / 2), y - h - int(pad / 2))
    rect_br = (x + w + int(pad / 2), y + int(pad / 2))
    
    cv2.rectangle(img, rect_tl, rect_br, bg_color, cv2.FILLED)
    cv2.putText(img, text, org, font, font_scale, text_color, thickness, cv2.LINE_AA)

def is_workstation_locked():
    hDesktop = ctypes.windll.user32.OpenInputDesktop(0, False, 0x0001)  # 0x0001 = DESKTOP_SWITCHDESKTOP
    if hDesktop == 0:
        return True
    ctypes.windll.user32.CloseDesktop(hDesktop)
    return False

def lock_workstation():
    if not is_workstation_locked():
        ctypes.windll.user32.LockWorkStation()
        print("Workstation locked")
    else:
        print("Could not lock workstation")

def check_finger(landmark_points, finger_tip_idx, finger_dip_idx, finger_pip_idx, finger_mcp_idx):
    finger_tip = landmark_points[finger_tip_idx]
    finger_dip = landmark_points[finger_dip_idx]
    finger_pip = landmark_points[finger_pip_idx]
    finger_mcp = landmark_points[finger_mcp_idx]
    if finger_tip[1] < finger_pip[1] and finger_tip[1] < finger_dip[1]:
        return True
    return False

def check_for_middle_finger(landmark_points):
    middle_up = check_finger(landmark_points, 12, 11, 10, 9)
    index_up = check_finger(landmark_points, 8, 7, 6, 5)
    ring_up = check_finger(landmark_points, 16, 15, 14, 13)
    pinky_up = check_finger(landmark_points, 20, 19, 18, 17)

    return middle_up and not index_up and not ring_up and not pinky_up

def draw_fingertip(image, fingertip_position):
    cv2.circle(image, fingertip_position, main_radius, main_color, cv2.FILLED)

capture = cv2.VideoCapture(0)

while True:
    success, image = capture.read()
    if not success:
        break

    height, width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmark_points = []
            for landmark in hand_landmarks.landmark:
                px, py = int(landmark.x * width), int(landmark.y * height)
                landmark_points.append((px, py))
                
            if len(landmark_points) >= 3:
                points = np.array(landmark_points, dtype=np.int32)
                hull = cv2.convexHull(points)
                
                overlay = image.copy()
                cv2.fillPoly(overlay, [hull], main_color)
                cv2.addWeighted(overlay, main_alpha, image, 1 - main_alpha, 0, image)
                cv2.polylines(image, [hull], isClosed=True, color=main_color, thickness=2)
                
            xs = [p[0] for p in landmark_points]
            ys = [p[1] for p in landmark_points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            box_tl = (max(0, x_min - main_pad), max(0, y_min - main_pad))
            box_br = (min(width, x_max + main_pad), min(height, y_max + main_pad))
            cv2.rectangle(image, box_tl, box_br, main_color, 2)
            
            if check_for_middle_finger(landmark_points):
                draw_text(image, "Middle Finger Up", (10, 60), bg_color=(0, 0, 0))
                if locking_enabled:
                    current_time = time.time()
                    if current_time - last_lock_time > debounce_time:
                        lock_workstation()
                        last_lock_time = current_time

            for fingertip_idx in [4, 8, 12, 16, 20]:
                draw_fingertip(image, landmark_points[fingertip_idx])

    if show_window:
        cv2.imshow("Hand Tracking", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()