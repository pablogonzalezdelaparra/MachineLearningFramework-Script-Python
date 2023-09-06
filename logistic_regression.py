# Importar las bibliotecas necesarias
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

# Cargar el conjunto de datos de California Housing.
# Este dataset contiene información sobre viviendas en California y 
# se convertirá en un problema de clasificación binaria.
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
X = housing.data
# Convertir a problema de clasificación binaria
y = (housing.target > 2.0).astype(int)

# Dividir el conjunto de datos en conjuntos de entrenamiento, 
# validación y prueba.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, 
                                                test_size=0.2, 
                                                random_state=42)

# Crear y entrenar un modelo de Regresión Logística.
model = LogisticRegression()
model.fit(X_train, y_train)

# Realizar predicciones en los conjuntos de validación y prueba.
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Calcular métricas de clasificación: precisión, recall, F1-score 
# y matriz de confusión.
# Estos parámetros nos indican lo siguiente:
# - Precisión: de todas las predicciones positivas, ¿cuántas 
# son realmente positivas?
# - Recall: de todas las etiquetas positivas, ¿cuántas fueron 
# correctamente predichas?
# - F1-score: promedio armónico entre precisión y recall.
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Imprimir los resultados de las métricas de clasificación.
print(f"Precisión: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
print(f"Matriz de Confusión:\n{conf_matrix}")

# Visualización de la matriz de confusión. Esta gráfica nos permite ver la 
# distribución de las predicciones.
sns.heatmap(conf_matrix, annot=True, cmap='Blues')
plt.xlabel("Predicciones")
plt.ylabel("Valores Reales")
plt.title("Matriz de Confusión")
plt.show()

# Visualización de la curva ROC y cálculo del área bajo la curva (AUC). 
# Esta gráfica nos permite ver la distribución de
# las predicciones en función del umbral de decisión.
y_val_prob = model.predict_proba(X_val)[:, 1]
fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
auc = roc_auc_score(y_val, y_val_prob)
plt.plot(fpr, tpr, linewidth=2, label=f"Curva ROC (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC")
plt.legend(loc="lower right")
plt.show()

# Visualización de la curva Precision-Recall. Esta gráfica nos permite ver 
# el progreso de la precisión y el recall
# en función del umbral de decisión.
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob)
plt.plot(recalls, precisions, linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall")
plt.show()

# Gráfico de barras para la distribución de etiquetas reales y predichas. 
# Esta gráfica nos permite ver la distribución
# de las etiquetas reales y predichas.
plt.bar([0, 1], [sum(y_val == 0), sum(y_val_pred == 0)], width=0.2, 
        label="Clase 0", align='center')
plt.bar([0.2, 1.2], [sum(y_val == 1), sum(y_val_pred == 1)], width=0.2, 
        label="Clase 1", align='center')
plt.xticks([0.1, 1.1], ["Reales", "Predichos"])
plt.xlabel("Clases")
plt.ylabel("Cantidad")
plt.title("Distribución de Clases (Reales vs. Predichas)")
plt.legend()
plt.show()
