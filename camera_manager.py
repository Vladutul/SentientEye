import cv2
import numpy as np # Adăugat opțional, doar pentru type hinting (vezi get_frame)
from picamera2 import Picamera2

# 1. Nume clar și specific pentru implementarea hardware
class PiCamera:
    def __init__(self, width: int = 640, height: int = 480, inverted_state: bool = False):
        print(f"[PiCamera] Inițializare la {width}x{height}...")
        self.inverted_state = inverted_state
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.picam2.configure(config)

    # 3. Type hints (-> None) pentru a indica faptul că nu returnează nimic
    def start(self) -> None:
        self.picam2.start()

    def stop(self) -> None:
        self.picam2.stop()

    # 2. Nume standardizat. Orice cameră vom adăuga pe viitor va avea metoda "get_frame"
    def get_frame(self) -> np.ndarray:
        """Returnează un frame gata de afișat (convertit în BGR și efect de oglindă)."""
        rgb_frame = self.picam2.capture_array("main")
        
        # Procesarea internă (specifică acestei camere) rămâne ascunsă aici
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if self.inverted_state:
            frame = self.mirror_frame(bgr_frame)
        else:
            frame = bgr_frame
        
        return frame
    
    def mirror_frame(self, frame):
        return cv2.flip(frame, 1)