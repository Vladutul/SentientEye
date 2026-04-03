import cv2
import numpy as np
from picamera2 import Picamera2

class PiCamera:
    def __init__(self, width: int = 640, height: int = 480, inverted_state: bool = False):
        print(f"[PiCamera] Inițializare la {width}x{height}...")
        self.inverted_state = inverted_state
        self.picam2 = Picamera2()
        
        # 💡 MODIFICARE: Cerem direct BGR888 pentru OpenCV (sau lasă RGB888 dacă folosești Web/Matplotlib)
        config = self.picam2.create_preview_configuration(
            main={"size": (width, height), "format": "BGR888"} 
        )
        self.picam2.configure(config)

    def start(self) -> None:
        self.picam2.start()

    def stop(self) -> None:
        self.picam2.stop()

    def get_frame(self) -> np.ndarray:
        # Array-ul vine acum direct în format BGR de la cameră
        frame = self.picam2.capture_array("main")

        if self.inverted_state:
            frame = self.mirror_frame(frame)
            
        return frame
    
    def mirror_frame(self, frame):
        return cv2.flip(frame, 1)