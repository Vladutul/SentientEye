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
        # Încercăm indexul cerut, dar dacă eșuează, testăm și 2 sau 5
        indices_de_testat = [self.device_index, 2, 5, 0, 1]
        
        for idx in indices_de_testat:
            print(f"Încerc să deschid camera pe /dev/video{idx}...")
            self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            
            if self.cap.isOpened():
                # Setăm formatul MJPG (esențial pentru camere Trust/USB 2.0)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                
                # Verificăm dacă putem citi un cadru real
                ret, frame = self.cap.read()
                if ret:
                    print(f"Succes! Cameră activă pe /dev/video{idx}")
                    self.device_index = idx
                    return
                else:
                    self.cap.release()
        
        raise RuntimeError("Nu am putut obține imagine de la niciun index video (1, 2 sau 5).")

    def stop(self) -> None:
        if self.cap:
            self.cap.release()

    def get_frame(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if self.inverted_state:
            frame = cv2.flip(frame, 1)
        return frame