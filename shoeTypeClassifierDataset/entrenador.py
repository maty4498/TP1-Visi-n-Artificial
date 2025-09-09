from sklearn import tree
from joblib import dump
import pandas as pd
import matplotlib.pyplot as plt

# cargar dataset generado en el paso 1
data = pd.read_csv("shoe_humoments.csv")  # columnas: h1...h7,label

# separar features (X) y etiquetas (Y)
X = data.iloc[:, :-1].values  # todas las columnas menos la última
Y = data.iloc[:, -1].values   # última columna = label

# entrenar clasificador
clasificador = tree.DecisionTreeClassifier(random_state=42)
clasificador.fit(X, Y)

# visualizar árbol (opcional)
plt.figure(figsize=(20,10))
tree.plot_tree(
    clasificador, 
    filled=True, 
    feature_names=[f"h{i}" for i in range(1,8)], 
    class_names=[
        "boots", "flip_flops", "loafers", 
        "sandals", "sneakers", "soccer_shoes"
    ]
)
plt.show()

# guardar modelo entrenado
dump(clasificador, "modelo_zapatos.joblib")
print("✅ Modelo entrenado y guardado como 'modelo_zapatos.joblib'")
