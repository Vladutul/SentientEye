import cv2
import time
from typing import Protocol, List, Dict, Any

class ICamera(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_frame(self) -> Any: ...

class IModelWorker(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def push_frame(self, frame: Any) -> None: ...
    def get_detections(self) -> List[Dict[str, Any]]: ...

class SentientEye:
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

        cv2.namedWindow("Sentient Eye", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Sentient Eye", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        try:
            while self.running_state:
                frame = self.camera.get_frame()
                
                if frame is not None:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                    self.model.push_frame(frame.copy())
                    detectii = self.model.get_detections()
                    self._draw_detections(frame, detectii)
                    
                    frame = cv2.resize(frame, (800, 480))
                    
                    cv2.imshow("Sentient Eye", frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    self.set_running_state(False)
                    break
        finally:
            self.cleanup()

    def _draw_detections(self, frame, detections):
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

if __name__ == "__main__":
    from camera_manager import PiCamera
    from ai_model_manager import YoloObjectDetector
    
    MODEL_PATH = "face_model_ncnn_model"
    
    my_camera = PiCamera(width=800, height=1000, inverted_state=True)
    my_model = YoloObjectDetector(model_path=MODEL_PATH, confidence_threshold=0.30)
    
    app = SentientEye(camera=my_camera, model=my_model)
    app.run()