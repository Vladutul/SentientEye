import cv2
import numpy as np

class USBCamera:
    def __init__(self, width: int = 640, height: int = 480, inverted_state: bool = False, device_index: int = 0):
        self.inverted_state = inverted_state
        self.device_index = device_index
        self.width = width
        self.height = height
        self.cap = None

    def start(self) -> None:
        # Inițializează camera USB
        self.cap = cv2.VideoCapture(self.device_index)
        
        # Setăm rezoluția dorită
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            raise RuntimeError(f"Nu s-a putut deschide camera USB cu indexul {self.device_index}")

    def stop(self) -> None:
        if self.cap:
            self.cap.release()

    def get_frame(self) -> np.ndarray:
        # ret este un boolean (True dacă a reușit să citească), frame este matricea imaginii
        ret, frame = self.cap.read()
        
        if not ret:
            # Returnează o matrice goală sau ridică o eroare dacă nu primește semnal
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if self.inverted_state:
            frame = self.mirror_frame(frame)
        
        return frame
    
    def mirror_frame(self, frame):
        # 1 înseamnă flip pe orizontală (oglindă)
        return cv2.flip(frame, 1)