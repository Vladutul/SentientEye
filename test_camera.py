import cv2

cap = cv2.VideoCapture(0) # Încearcă 0, apoi 1, apoi 2
if not cap.isOpened():
    print("Nu pot deschide camera sub nicio formă.")
else:
    print("Camera e deschisă! Apasă 'q' pentru a închide fereastra.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow("Test Direct", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()