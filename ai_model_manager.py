import threading
import queue
import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO
import RPi.GPIO as GPIO
import time

class YoloObjectDetector:
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.20, buzzer_pin: int = 13):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.frame_queue = queue.Queue(maxsize=1)
        self.current_detections: List[Dict[str, Any]] = []
        self.running_state = False
        self.thread: threading.Thread | None = None
        
        # --- NOU: Flag pentru a preveni spam-ul de thread-uri pe buzzer ---
        self.is_buzzing = False 

        # Setup GPIO buzzer
        GPIO.setmode(GPIO.BCM)
        self.buzzer_pin = buzzer_pin
        GPIO.setup(self.buzzer_pin, GPIO.OUT)
        self.pwm_signal = GPIO.PWM(self.buzzer_pin, 1000)
        self.pwm_signal.start(0)  # 0% duty cycle (buzzer off)

    def start(self) -> None:
        self.change_running_state(True)
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.change_running_state(False)
        if self.thread is not None:
            self.thread.join()
        self.pwm_signal.stop()
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
                    
                    # --- MODIFICARE CRITICĂ AICI ---
                    # Setăm conf=0.20 ca să nu mai filtreze scorurile de 0.30 sau 0.40 înainte să le vedem
                    results = model(frame, conf=0.20, verbose=False)

                    new_detection = []
                    for r in results:
                        for box in r.boxes:
                            confidence_score = float(box.conf[0])
                            nume_obiect = r.names[int(box.cls[0])]

                            # --- AFIȘĂRI DISTINCTE PENTRU OPEN ȘI CLOSE ---
                            if nume_obiect == "open":
                                print(f"🟢 [DETECȚIE] OPEN detectat | Încredere: {confidence_score:.2f}")
                            elif nume_obiect == "close":
                                print(f"🔴 [DETECȚIE] CLOSE detectat | Încredere: {confidence_score:.2f}")
                            else:
                                print(f"⚪ [DETECȚIE] {nume_obiect} | Încredere: {confidence_score:.2f}")

                            new_detection.append({
                                "nume": nume_obiect,
                                "coord": box.xyxy[0].cpu().numpy().astype(int),
                                "confidence": confidence_score
                            })

                            # Buzzer-ul se activează DOAR pentru "close" cu încredere > 0.40
                            if nume_obiect == "close" and confidence_score > 0.40:
                                self._buzz_for_duration(1.0)

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

    def _buzz_for_duration(self, duration: float):
        """Activează buzzerul doar dacă nu este deja activ, pentru a preveni blocajele de CPU."""
        
        # --- NOU: Dacă buzzerul deja sună, ignorăm comanda nouă ---
        if self.is_buzzing:
            return 
            
        def beep_task():
            self.is_buzzing = True
            self.pwm_signal.ChangeDutyCycle(50)  # Pornim buzzerul
            time.sleep(duration)
            self.pwm_signal.ChangeDutyCycle(0)   # Oprim buzzerul
            self.is_buzzing = False              # Eliberăm flag-ul
            
        # Lansăm funcția de beep într-un thread separat
        threading.Thread(target=beep_task, daemon=True).start()