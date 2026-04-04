import cv2
import numpy as np
from picamera2 import Picamera2

class PiCamera:
    def __init__(self, width: int = 640, height: int = 480, inverted_state: bool = False):
        self.inverted_state = inverted_state
        self.picam2 = Picamera2()
        
        config = self.picam2.create_preview_configuration(
            main={"size": (width, height), "format": "BGR888"} 
        )
        self.picam2.configure(config)

    def start(self) -> None:
        self.picam2.start()
        
        # --- 💡 MODIFICAREA CRUCIALĂ PENTRU CAMERA ROȘIE ---
        # Dezactivăm balansul de alb automat care eșuează pe senzorul tău
        # Valorile (Red, Blue): Scădem Roșu la 0.5 (jumătate) și creștem Albastru la 2.5
        # Joacă-te cu aceste cifre dacă încă e prea roșu (ex: 0.3 în loc de 0.5)
        self.picam2.set_controls({"AwbEnable": False, "ColourGains": (0.5, 2.5)})
        # --------------------------------------------------

    def stop(self) -> None:
        self.picam2.stop()

    def get_frame(self) -> np.ndarray:
        frame = self.picam2.capture_array("main")

        if self.inverted_state:
            frame = self.mirror_frame(frame)
        
        return frame
    
    def mirror_frame(self, frame):
        return cv2.flip(frame, 1)