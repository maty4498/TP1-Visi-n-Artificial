import cv2
import numpy as np
from joblib import load

# Diccionario de etiquetas (usar los mismos números del entrenamiento)
label_dict = {
    0: "Boots",
    1: "Flip Flops",
    2: "Loafers",
    3: "Sandals",
    4: "Sneakers",
    5: "Soccer Shoes"
}

# Cargar modelo entrenado
clasificador = load("modelo_zapatos.joblib")

# Abrir webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:  # filtrar ruido
            huMoments = cv2.HuMoments(cv2.moments(cnt)).flatten().reshape(1, -1)
            pred = clasificador.predict(huMoments)[0]

            # Dibujar rectángulo y etiqueta
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(frame, label_dict.get(pred, "Desconocido"), 
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0,0,255), 2)

    cv2.imshow("Clasificador Zapatos", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
