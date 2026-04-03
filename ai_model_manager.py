import threading
import queue
import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO
import RPi.GPIO as GPIO
import time

class YoloObjectDetector:
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.50, buzzer_pin: int = 13):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.frame_queue = queue.Queue(maxsize=1)
        self.current_detections: List[Dict[str, Any]] = []
        self.running_state = False
        self.thread: threading.Thread | None = None

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
        model = YOLO(self.model_path)
        print("[YOLO] Pregătit pentru analiză!")

        while self.running_state:
            try:
                frame = self.frame_queue.get(timeout=1)
                results = model(frame, conf=self.confidence_threshold, verbose=False)

                new_detection = []
                for r in results:
                    for box in r.boxes:
                        confidence_score = float(box.conf[0])
                        new_detection.append({
                            "nume": r.names[int(box.cls[0])],
                            "coord": box.xyxy[0].cpu().numpy().astype(int),
                            "confidence": confidence_score
                        })

                        # Dacă detectează ceva cu confidence > 0.6 → buzzer 1 sec
                        if confidence_score > 0.6:
                            self._buzz_for_duration(1.0)

                self.current_detections = new_detection

            except queue.Empty:
                continue

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
        """Activează buzzerul pentru o perioadă scurtă pe un thread separat pentru a nu bloca detectia YOLO."""
        def beep_task():
            self.pwm_signal.ChangeDutyCycle(50)  # Pornim buzzerul
            time.sleep(duration)
            self.pwm_signal.ChangeDutyCycle(0)   # Oprim buzzerul
            
        # Lansăm funcția de beep într-un thread separat (nu blochează bucla principală)
        threading.Thread(target=beep_task, daemon=True).start()