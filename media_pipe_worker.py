import cv2
import mediapipe as mp
import math
from typing import List, Dict, Any

class MediaPipeWorker:
    def __init__(self, ear_threshold: float = 0.22, consecutive_frames: int = 10):
        """
        :param ear_threshold: Valoarea sub care considerăm ochiul închis.
        :param consecutive_frames: Câte cadre la rând trebuie să aibă ochiul închis ca să declanșeze starea "ADORMIT".
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        # Inițializăm modelul de Face Mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.current_detections = []
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.sleep_frames = 0

        # Punctele specifice pentru conturul ochilor în MediaPipe
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def start(self) -> None:
        print("[MediaPipeWorker] Model încărcat și pregătit.")

    def stop(self) -> None:
        self.face_mesh.close()
        print("[MediaPipeWorker] Model oprit.")

    def _euclidean_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _calculate_ear(self, landmarks, eye_indices):
        # Extragem coordonatele pentru cele 6 puncte ale ochiului
        p1 = landmarks.landmark[eye_indices[0]]
        p2 = landmarks.landmark[eye_indices[1]]
        p3 = landmarks.landmark[eye_indices[2]]
        p4 = landmarks.landmark[eye_indices[3]]
        p5 = landmarks.landmark[eye_indices[4]]
        p6 = landmarks.landmark[eye_indices[5]]

        # Distanțele verticale
        v1 = self._euclidean_distance(p2, p6)
        v2 = self._euclidean_distance(p3, p5)
        # Distanța orizontală
        h = self._euclidean_distance(p1, p4)

        if h == 0:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    def push_frame(self, frame: Any) -> None:
        self.current_detections = []
        
        # MediaPipe necesită imagini în format RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesăm cadrul
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_ear = self._calculate_ear(face_landmarks, self.LEFT_EYE)
                right_ear = self._calculate_ear(face_landmarks, self.RIGHT_EYE)
                
                # Media dintre ochiul stâng și cel drept
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Logica de somnolență
                if avg_ear < self.ear_threshold:
                    self.sleep_frames += 1
                else:
                    self.sleep_frames = 0
                    
                if self.sleep_frames >= self.consecutive_frames:
                    stare = "!!! ADORMIT !!!"
                else:
                    stare = f"Treaz (EAR: {avg_ear:.2f})"
                
                # Pentru compatibilitate cu self._draw_detections din SentientEye,
                # generăm un Bounding Box care să încadreze toată fața.
                h, w, _ = frame.shape
                x_min = int(min([lm.x for lm in face_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in face_landmarks.landmark]) * h)
                x_max = int(max([lm.x for lm in face_landmarks.landmark]) * w)
                y_max = int(max([lm.y for lm in face_landmarks.landmark]) * h)
                
                # Salvăm detecția în formatul cerut de interfață
                self.current_detections.append({
                    "coord": [max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max)],
                    "nume": stare
                })

    def get_detections(self) -> List[Dict[str, Any]]:
        return self.current_detections