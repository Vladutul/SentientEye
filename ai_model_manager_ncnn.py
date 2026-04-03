import threading
import queue
import numpy as np
import cv2
from typing import List, Dict, Any
import ncnn
import RPi.GPIO as GPIO
import time

class NcnnYoloDetector:
    def __init__(self, 
                 param_path: str = "model.param", 
                 bin_path: str = "model.bin", 
                 input_size: int = 640, # Dimensiunea la care a fost antrenat modelul (ex: 640 pt YOLO)
                 confidence_threshold: float = 0.20, 
                 buzzer_pin: int = 13):
        
        self.param_path = param_path
        self.bin_path = bin_path
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        
        # Maparea claselor (NCNN returnează doar ID-ul clasei ca număr)
        # Trebuie să le pui în ordinea exactă în care au fost antrenate!
        self.class_names = {0: "close", 1: "open"} # Modifică cu clasele tale reale
        
        self.frame_queue = queue.Queue(maxsize=1)
        self.current_detections: List[Dict[str, Any]] = []
        self.running_state = False
        self.thread: threading.Thread | None = None
        
        self.is_buzzing = False 

        # Setup GPIO buzzer
        GPIO.setmode(GPIO.BCM)
        self.buzzer_pin = buzzer_pin
        GPIO.setup(self.buzzer_pin, GPIO.OUT)
        self.pwm_signal = GPIO.PWM(self.buzzer_pin, 1000)
        self.pwm_signal.start(0)  

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
        print("[NCNN] Încarc modelul pe nucleele hardware...")
        try:
            net = ncnn.Net()
            # Opțional: Activează Vulkan dacă RPi-ul tău suportă/este compilat pentru asta
            net.opt.use_vulkan_compute = False 
            
            net.load_param(self.param_path)
            net.load_model(self.bin_path)
            print("[NCNN] Pregătit pentru analiză!")
        except Exception as e:
            print(f"[EROARE FATALĂ NCNN] Modelul nu s-a putut încărca: {e}")
            return

        while self.running_state:
            try:
                frame = self.frame_queue.get(timeout=1)
                h_orig, w_orig = frame.shape[:2]

                # 1. PREGĂTIREA IMAGINII PENTRU NCNN (YOLO standard)
                mat_in = ncnn.Mat.from_pixels_resize(
                    frame,
                    ncnn.Mat.PixelType.PIXEL_BGR2RGB,
                    w_orig, h_orig,
                    self.input_size, self.input_size
                )
                
                # Normalizare specifică YOLOv8 (pixelii împărțiți la 255)
                norm_vals = [1/255.0, 1/255.0, 1/255.0]
                mat_in.substract_mean_normalize([], norm_vals)

                # 2. INFERENȚA
                ex = net.create_extractor()
                
                ex.input("in0", mat_in) 
                ret, mat_out = ex.extract("out0")

                # 3. PROCESAREA REZULTATELOR
                new_detection = []
                
                # Cazul ideal: modelul returnează o matrice de tip [num_detectii, 6] 
                # (label, confidence, x1, y1, x2, y2)
                if mat_out is not None and mat_out.h > 0:
                    for i in range(mat_out.h):
                        values = mat_out.row(i)
                        
                        label_id = int(values[0])
                        confidence_score = float(values[1])
                        
                        if confidence_score < self.confidence_threshold:
                            continue

                        # Coordonatele vin raportate la dimensiunea de intrare (self.input_size)
                        # Trebuie să le scalăm înapoi la rezoluția camerei (w_orig, h_orig)
                        x1 = int(values[2] * w_orig / self.input_size)
                        y1 = int(values[3] * h_orig / self.input_size)
                        x2 = int(values[4] * w_orig / self.input_size)
                        y2 = int(values[5] * h_orig / self.input_size)

                        nume_obiect = self.class_names.get(label_id, f"Clasa_{label_id}")

                        # --- AFIȘĂRI DISTINCTE ---
                        #if nume_obiect == "open":
                        #    print(f"🟢 [NCNN] OPEN detectat | Încredere: {confidence_score:.2f}")
                        #elif nume_obiect == "close":
                        #    print(f"🔴 [NCNN] CLOSE detectat | Încredere: {confidence_score:.2f}")
                        #else:
                        #    print(f"⚪ [NCNN] {nume_obiect} | Încredere: {confidence_score:.2f}")

                        new_detection.append({
                            "nume": nume_obiect,
                            "coord": (x1, y1, x2, y2),
                            "confidence": confidence_score
                        })

                        # Buzzer-ul se activează DOAR pentru "close" cu încredere > 0.40
                        if nume_obiect == "close" and confidence_score > 0.40:
                            self._buzz_for_duration(1.0)

                self.current_detections = new_detection

            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n[EROARE ÎN THREAD-UL DE ANALIZĂ NCNN] -> {e}\n")

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
        if self.is_buzzing:
            return 
            
        def beep_task():
            self.is_buzzing = True
            self.pwm_signal.ChangeDutyCycle(50)  
            time.sleep(duration)
            self.pwm_signal.ChangeDutyCycle(0)   
            self.is_buzzing = False              
            
        threading.Thread(target=beep_task, daemon=True).start()