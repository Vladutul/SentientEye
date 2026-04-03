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

                # --- 3. PROCESAREA REZULTATELOR (DECODOR YOLOv8) ---
                new_detection = []
                
                if mat_out is not None:
                    # Convertim ieșirea brută în matrice numpy (Forma tipică: 6 rânduri, 8400 coloane)
                    out = np.array(mat_out)
                    
                    # Dacă matricea e "culcată" (6, 8400), o întoarcem (8400, 6) pentru a o citi ușor
                    if len(out.shape) == 2 and out.shape[0] < out.shape[1]:
                        out = out.T
                        
                    if len(out) > 0:
                        # Extragem datele
                        boxes = out[:, :4]  # Primele 4 coloane: cx, cy, w, h
                        scores = out[:, 4:] # Următoarele coloane: încrederea pt "close" și "open"
                        
                        # Găsim clasa câștigătoare pentru fiecare din cele 8400 de cutii
                        class_ids = np.argmax(scores, axis=1)
                        max_scores = np.max(scores, axis=1)
                        
                        # Filtrăm DOAR cutiile care depășesc pragul de încredere setat de tine
                        mask = max_scores > self.confidence_threshold
                        filtered_boxes = boxes[mask]
                        filtered_scores = max_scores[mask]
                        filtered_class_ids = class_ids[mask]
                        
                        # Transformăm din (Centru_X, Centru_Y, Lățime, Înălțime) în variabile separate
                        cx = filtered_boxes[:, 0]
                        cy = filtered_boxes[:, 1]
                        w = filtered_boxes[:, 2]
                        h = filtered_boxes[:, 3]
                        
                        # Scalăm coordonatele înapoi la rezoluția camerei tale (w_orig, h_orig)
                        x1_arr = (cx - w / 2) * (w_orig / self.input_size)
                        y1_arr = (cy - h / 2) * (h_orig / self.input_size)
                        w_arr = w * (w_orig / self.input_size)
                        h_arr = h * (h_orig / self.input_size)
                        
                        # Aplicăm NMS (Non-Maximum Suppression) pentru a șterge cutiile dublate
                        bboxes_for_nms = []
                        for i in range(len(x1_arr)):
                            bboxes_for_nms.append([int(x1_arr[i]), int(y1_arr[i]), int(w_arr[i]), int(h_arr[i])])
                            
                        indices = cv2.dnn.NMSBoxes(
                            bboxes=bboxes_for_nms,
                            scores=filtered_scores.tolist(),
                            score_threshold=self.confidence_threshold,
                            nms_threshold=0.45  # Elimină suprapunerile mai mari de 45%
                        )
                        
                        # Dacă au rămas cutii valide după NMS, le trimitem la desenat!
                        if len(indices) > 0:
                            for idx in indices.flatten():
                                final_x1 = int(x1_arr[idx])
                                final_y1 = int(y1_arr[idx])
                                final_x2 = final_x1 + int(w_arr[idx])
                                final_y2 = final_y1 + int(h_arr[idx])
                                
                                score = filtered_scores[idx]
                                class_id = filtered_class_ids[idx]
                                nume_obiect = self.class_names.get(class_id, f"Clasa_{class_id}")
                                
                                new_detection.append({
                                    "nume": f"{nume_obiect} {score:.2f}",
                                    "coord": (final_x1, final_y1, final_x2, final_y2),
                                    "confidence": score
                                })
                                
                                # Logica ta de buzzer
                                if nume_obiect == "close" and score > 0.40:
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