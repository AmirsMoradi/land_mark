import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)



class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    def predict(self, coord):
        self.kf.correct(np.array(coord, np.float32))
        pred = self.kf.predict()
        return int(pred[0]), int(pred[1])


kf_dict = {}


cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
        mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
        mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("عدم دریافت فریم!")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_pose = pose.process(frame_rgb)
        results_hands = hands.process(frame_rgb)
        results_face = face_mesh.process(frame_rgb)

        annotated_frame = frame.copy()

        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(annotated_frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results_pose.pose_landmarks.landmark
            joints = {
                "Left Knee": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE,
                              mp_pose.PoseLandmark.LEFT_ANKLE],
                "Right Knee": [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE,
                               mp_pose.PoseLandmark.RIGHT_ANKLE],
                "Left Elbow": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW,
                               mp_pose.PoseLandmark.LEFT_WRIST],
                "Right Elbow": [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW,
                                mp_pose.PoseLandmark.RIGHT_WRIST],
                "Neck": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.NOSE,
                         mp_pose.PoseLandmark.RIGHT_SHOULDER],
                "Hip": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE]
            }

            for joint, points in joints.items():
                try:
                    a = [landmarks[points[0]].x, landmarks[points[0]].y]
                    b = [landmarks[points[1]].x, landmarks[points[1]].y]
                    c = [landmarks[points[2]].x, landmarks[points[2]].y]
                    angle = calculate_angle(a, b, c)
                    pos = tuple(np.multiply(b, [640, 480]).astype(int))

                    if joint not in kf_dict:
                        kf_dict[joint] = KalmanFilter()
                    filtered_pos = kf_dict[joint].predict(pos)

                    cv2.putText(annotated_frame, f'{joint}: {int(angle)}°', filtered_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    print(f"خطا در محاسبه زاویه {joint}: {e}")

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3))

        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                try:
                    mask = np.zeros_like(frame, dtype=np.uint8)
                    points = [(int(l.x * frame.shape[1]), int(l.y * frame.shape[0])) for l in face_landmarks.landmark]
                    cv2.fillConvexPoly(mask, np.array(points, dtype=np.int32), (255, 255, 255))
                    annotated_frame = cv2.addWeighted(annotated_frame, 1, mask, 0.3, 0)
                except KeyError as e:
                    print(f"خطای لندمارک صورت: {e}")

        cv2.imshow("Original Frame", frame)
        cv2.imshow("Processed Landmarks", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
