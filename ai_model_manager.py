import threading
import queue
import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO
import RPi.GPIO as GPIO
import time
from send_open_or_closed import trimite_stare # Importul tău

class YoloObjectDetector:
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.50):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.frame_queue = queue.Queue(maxsize=1)
        self.current_detections: List[Dict[str, Any]] = []
        self.running_state = False
        self.thread: threading.Thread | None = None
        
        self.ultima_stare_trimisa = None 

    def start(self) -> None:
        self.change_running_state(True)
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.change_running_state(False)
        if self.thread is not None:
            self.thread.join()
        GPIO.cleanup()

    def _worker_loop(self) -> None:
            print("[YOLO] Încarc modelul pe nucleele hardware...")
            try:
                model = YOLO(self.model_path)
                print("[YOLO] Pregătit pentru analiză!")
            except Exception as e:
                print(f"[EROARE FATALĂ YOLO] Modelul nu s-a putut încărca: {e}")
                return

            while self.running_state:
                try:
                    frame = self.frame_queue.get(timeout=1)
                    results = model(frame, conf=0.50, verbose=False)

                    new_detection = []
                    # Variabilă temporară pentru a vedea ce am găsit în acest cadru
                    stare_curenta_detectata = None

                    for r in results:
                        for box in r.boxes:
                            confidence_score = float(box.conf[0])
                            nume_obiect = r.names[int(box.cls[0])]

                            if nume_obiect == "open":
                                print(f"🟢 [DETECȚIE] OPEN | Conf: {confidence_score:.2f}")
                                stare_curenta_detectata = "1"
                            elif nume_obiect == "close":
                                print(f"🔴 [DETECȚIE] CLOSE | Conf: {confidence_score:.2f}")
                                stare_curenta_detectata = "0"

                            new_detection.append({
                                "nume": nume_obiect,
                                "coord": box.xyxy[0].cpu().numpy().astype(int),
                                "confidence": confidence_score
                            })

                    if stare_curenta_detectata is not None and stare_curenta_detectata != self.ultima_stare_trimisa:
                        try:
                            trimite_stare(stare_curenta_detectata)
                            self.ultima_stare_trimisa = stare_curenta_detectata
                            print(f"📡 [NETWORK] Flag {stare_curenta_detectata} trimis prin Ethernet.")
                        except Exception as e:
                            print(f"⚠️ [EROARE REȚEA] Nu s-a putut trimite: {e}")

                    self.current_detections = new_detection

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"\n[EROARE ÎN THREAD-UL DE ANALIZĂ] -> {e}\n")

    def push_frame(self, frame: np.ndarray) -> None:
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    def get_detections(self) -> List[Dict[str, Any]]:
        return self.current_detections

    def change_running_state(self, state: bool):
        self.running_state = state