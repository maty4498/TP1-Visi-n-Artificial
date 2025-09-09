import cv2
import os
import numpy as np
import pandas as pd

# Ruta a tu dataset (cambiar si es necesario)
dataset_dir = r"D:/Mis Descargas/shoeTypeClassifierDataset/training"

# Mapeo de etiquetas a números
class_map = {
    "boots": 0,
    "flip_flops": 1,
    "loafers": 2,
    "sandals": 3,
    "sneakers": 4,
    "soccer_shoes": 5
}

data = []
labels = []

for class_name, class_id in class_map.items():
    class_dir = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    
    for fname in os.listdir(class_dir):
        path = os.path.join(class_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Preprocesamiento
        blur = cv2.GaussianBlur(img, (5,5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        edges = cv2.Canny(thresh, 50, 150)

        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Tomar el contorno más grande
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 100:  # descartar ruido muy chico
            continue

        # Calcular Hu Moments
        hu = cv2.HuMoments(cv2.moments(cnt)).flatten()

        # Guardar en dataset
        data.append(hu)
        labels.append(class_id)

# Crear DataFrame y guardar en CSV
df = pd.DataFrame(data, columns=[f"h{i}" for i in range(1,8)])
df["label"] = labels
df.to_csv("shoe_humoments.csv", index=False)

print("✅ Dataset guardado con", len(df), "muestras en shoe_humoments.csv")
