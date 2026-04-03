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
        
        # Un mic delay pentru a lăsa camera să se stabilizeze (auto-exposure)
        time.sleep(2)
        
        cv2.namedWindow("Sentient Eye", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Sentient Eye", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        try:
            while self.running_state:
                frame = self.camera.get_frame()
                
                if frame is None:
                    continue

                # 1. Rotația - esențială dacă camera e montată fizic la 90 grade
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                # 2. AI Inference
                # Notă: Asigură-te că NcnnYoloDetector redimensionează intern la 320/640
                self.model.push_frame(frame) 
                detectii = self.model.get_detections()
                
                # 3. Desenăm pe frame-ul de rezoluție mare (pentru acuratețea liniilor)
                if detectii:
                    self._draw_detections(frame, detectii)
                
                # 4. Resize final pentru display-ul de 800x480
                display_frame = cv2.resize(frame, (800, 480))
                
                cv2.imshow("Sentient Eye", display_frame)

                if cv2.waitKey(1) & 0xFF == 27: # ESC
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
    from camera_manager import PiCamera 
    from ai_model_manager_ncnn import NcnnYoloDetector  # <-- Importăm noua clasă
    
    # 1. Punem calea EXACTĂ către cele două fișiere din folderul generat de Ultralytics
    PARAM_PATH = "face_model_ncnn_model/model.ncnn.param" 
    BIN_PATH = "face_model_ncnn_model/model.ncnn.bin"
    
    # 2. Instanțiem camera
    my_camera = PiCamera(width=1920, height=1080, inverted_state=True)
    
    # 3. Instanțiem modelul brut NCNN
    my_model = NcnnYoloDetector(
        param_path=PARAM_PATH, 
        bin_path=BIN_PATH, 
        input_size=640,  # Pune 320 aici dacă ai exportat modelul la 320x320 cum am discutat!
        confidence_threshold=0.50,
        buzzer_pin=13
    )
    
    # 4. Injectăm și rulăm aplicația (Nu se schimbă absolut nimic în logica SentientEye!)
    app = SentientEye(camera=my_camera, model=my_model)
    app.run()