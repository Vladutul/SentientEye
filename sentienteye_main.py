import cv2
import time
from typing import Protocol, List, Dict, Any

# 1. Definim "Contractele" (Interfețele)
# Orice cameră viitoare va trebui să respecte această structură
class ICamera(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_frame(self) -> Any: ... # Returnează framel (ex: BGR flipped)

# Orice model de AI viitor va trebui să respecte această structură
class IModelWorker(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def push_frame(self, frame: Any) -> None: ...
    def get_detections(self) -> List[Dict[str, Any]]: ...

# 2. Clasa principală devine complet independentă de device-uri
class SentientEye:
    # Injectăm dependințele prin constructor
    def __init__(self, camera: ICamera, model: IModelWorker):
        self.camera = camera
        self.model = model
        self.running_state = False

    def run(self):
        print("--- START: Pregătire Unelte ---")
        self.start_components()
        self.set_running_state(True)
        
        time.sleep(3)
        print("--- RUNTIME: Sistemul rulează (Apasă ESC pentru ieșire) ---")

        # 1. Creăm fereastra și o setăm pe Full Screen chiar de la început
        cv2.namedWindow("Sentient Eye", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Sentient Eye", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        try:
            while self.running_state:
                # framel vine deja procesat și orientat corect din obiectul camerei
                frame = self.camera.get_frame()
                
                if frame is not None:
                    # Dacă ecranul tău stă pe orizontală (landscape), s-ar putea să nu mai ai 
                    # nevoie de rotația de 90 de grade. Dacă stă pe verticală, las-o.
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                    # Trimitem frame-ul la modelul AI înainte de resize 
                    # (pentru ca AI-ul să proceseze imaginea la calitatea ei originală)
                    self.model.push_frame(frame.copy())
                    detectii = self.model.get_detections()
                    self._draw_detections(frame, detectii)
                    
                    # --- STRETCH PE TOATĂ SUPRAFAȚA ECRANULUI ---
                    # Forțăm rezoluția la 800 lățime și 480 înălțime
                    frame = cv2.resize(frame, (800, 480))
                    
                    cv2.imshow("Sentient Eye", frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    self.set_running_state(False)
                    break
        finally:
            self.cleanup()

    def _draw_detections(self, frame, detections):
        """O metodă separată doar pentru desenare (Single Responsibility)"""
        for obiect in detections:
            x1, y1, x2, y2 = obiect["coord"]
            nume = obiect["nume"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, nume, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def start_components(self):
        self.model.start()
        self.camera.start()

    def stop_components(self):
        self.model.stop()
        self.camera.stop()

    def set_running_state(self, state: bool):
        self.running_state = state

    def cleanup(self):
        print("Încep oprirea componentelor...")
        self.stop_components()
        cv2.destroyAllWindows()
        print("Sistem oprit cu succes.")

# 3. Asamblarea aplicației se face la exterior (Compozitie)
if __name__ == "__main__":
    from camera_manager import PiCamera # Implementarea concretă
    from ai_model_manager import YoloObjectDetector     # Implementarea concretă
    
    MODEL_PATH = "face_model_ncnn_model"

    # Aici poți schimba ușor cu:
    # camera = IpCameraManager("192.168.1.100")
    # model = MediaPipeWorker()
    
    my_camera = PiCamera(width=1200, height=600, inverted_state=True)
    my_model = YoloObjectDetector(model_path=MODEL_PATH, confidence_threshold=0.50)
    
    app = SentientEye(camera=my_camera, model=my_model)
    app.run()