import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar solo objetos confiables y clases específicas
    results = model(frame, conf=0.6, classes=[0, 2, 16])  # persona, carro, perro

    # Dibujar cajas y etiquetas
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
