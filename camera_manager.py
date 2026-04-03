import cv2
import numpy as np
from picamera2 import Picamera2

class PiCamera:
    def __init__(self, width: int = 640, height: int = 480, inverted_state: bool = False):
        self.inverted_state = inverted_state
        self.picam2 = Picamera2()
        
        # 1. Schimbăm formatul de configurare în BGR888
        # OpenCV folosește BGR nativ. Dacă trimiți RGB, albastrul devine roșu/portocaliu.
        config = self.picam2.create_preview_configuration(
            main={"size": (width, height), "format": "BGR888"} 
        )
        self.picam2.configure(config)

    def start(self) -> None:
        self.picam2.start()

    def stop(self) -> None:
        self.picam2.stop()

    def get_frame(self) -> np.ndarray:
        # 2. Acum capture_array va returna direct formatul BGR
        frame = self.picam2.capture_array("main")
        
        # ELIMINĂ linia veche: cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        # Deoarece acum am setat camera să scoată direct BGR de la sursă.

        if self.inverted_state:
            frame = self.mirror_frame(frame)
        
        return frame
    
    def mirror_frame(self, frame):
        return cv2.flip(frame, 1)