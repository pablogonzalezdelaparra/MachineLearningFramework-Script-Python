# Importar las bibliotecas necesarias
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, confusion_matrix
import seaborn as sns
import numpy as np
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

accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
conf_matrix = confusion_matrix(y_val, y_val_pred)

# Imprimir los resultados de las métricas de clasificación.
print(f"Precisión: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
print(f"Matriz de Confusión:\n{conf_matrix}")

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

# Graficar la distribucion de las etiquetas reales y predichas.
plt.bar([0, 1], [sum(y_test == 0), sum(y_test_pred == 0)], width=0.2,
        label="Clase 0", align='center')
plt.bar([0.2, 1.2], [sum(y_test == 1), sum(y_test_pred == 1)], width=0.2,
        label="Clase 1", align='center')
plt.xticks([0.1, 1.1], ["Reales", "Predichos"])
plt.xlabel("Clases")
plt.ylabel("Cantidad")
plt.title("Distribución de Clases (Reales vs. Predichas)")
plt.legend()
plt.savefig("distribucion_clases.png")

# Calcular la distribucion de las etiquetas reales y predichas.
print(f"Etiquetas reales: {sum(y_test == 0)} de clase 0 y {sum(y_test == 1)} de clase 1")
print(f"Etiquetas predichas: {sum(y_test_pred == 0)} de clase 0 y {sum(y_test_pred == 1)} de clase 1")

# Calcular el porcentaje de la distribucion de las etiquetas reales y predichas.
print(f"Etiquetas reales: {sum(y_test == 0) / len(y_test) * 100:.2f}% de clase 0 y {sum(y_test == 1) / len(y_test) * 100:.2f}% de clase 1")
print(f"Etiquetas predichas: {sum(y_test_pred == 0) / len(y_test_pred) * 100:.2f}% de clase 0 y {sum(y_test_pred == 1) / len(y_test_pred) * 100:.2f}% de clase 1")

# graficar la diferencia entre y_test y y_test_pred
plt.bar([0, 1], [sum(y_test == 0) - sum(y_test_pred == 0), sum(y_test == 1) - sum(y_test_pred == 1)], width=0.2,
        label="Clase 0", align='center')
plt.xticks([0, 1], ["Clase 0", "Clase 1"])
plt.xlabel("Clases")
plt.ylabel("Cantidad")
plt.title("Diferencia entre etiquetas reales y predichas")
plt.legend()
plt.show()

# dividir el conjunto de test en 10 partes con el modelo ya entrenado, y calcular la precisión de cada parte para mostrar la varianza del modelo
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
model = LogisticRegression()
model.fit(X_train, y_train)
scores = []
for train_index, test_index in kf.split(X_test):
        scores.append(model.score(X_test[test_index], y_test[test_index])) 
print(f"Test scores: {scores}")     
print(f"Mean scores after 10 runs:")
print(f"Test score: {np.mean(scores):.6f}")

# graficar la diferencia de los scores de cada parte del conjunto de test para ver la varianza del modelo
plt.bar(range(10), scores, width=0.2,
        label="Clase 0", align='center')
plt.xticks(range(10), range(10))
plt.xlabel("Partes")
plt.ylabel("Precisión")
plt.title("Precisión de cada parte del conjunto de test")
plt.legend()
plt.savefig("precision_test.png")





"""
# Print scores of x_train, x_val and x_test
print(f"Train score: {model.score(X_train, y_train):.6f}")
print(f"Validation score: {model.score(X_val, y_val):.6f}")
print(f"Test score: {model.score(X_test, y_test):.6f}")

# run the model 10 times and get the average score for x_train, x_val and x_test
train_scores = []
val_scores = []
test_scores = []
for i in range(10):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        train_scores.append(model.score(X_train, y_train))
        val_scores.append(model.score(X_val, y_val))
        test_scores.append(model.score(X_test, y_test))
print("Mean scores after 10 runs:")
print(f"Train score: {np.mean(train_scores):.6f}")
print(f"Validation score: {np.mean(val_scores):.6f}")
print(f"Test score: {np.mean(test_scores):.6f}")

# do a cross validation with 10 folds
from sklearn.model_selection import cross_val_score
scores = cross_val_score(LogisticRegression(), X, y, cv=10)
print(f"Cross-validation scores: {scores}")
print(f"Mean cross-validation score: {np.mean(scores):.6f}")


# create graph learning curve
from sklearn.model_selection import learning_curve
train_sizes, train_scores, val_scores = learning_curve(LogisticRegression(), X, y, cv=15)
plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Training score")
plt.plot(train_sizes, np.mean(val_scores, axis=1), label="Validation score")
plt.xlabel("Training set size")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("learning_curve.png")
"""

"""
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
"""

# Métodos de regularización
"""
# Crear y entrenar un modelo de Regresión Logística sin regularización (lr0).
model_lr0 = LogisticRegression(penalty='none', random_state=42)
model_lr0.fit(X_train, y_train)

# Crear y entrenar un modelo de Regresión Logística con regularización L1 (lr1).
model_lr1 = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
model_lr1.fit(X_train, y_train)

# Crear y entrenar un modelo de Regresión Logística con regularización L2 (lr2).
model_lr2 = LogisticRegression(penalty='l2', random_state=42)
model_lr2.fit(X_train, y_train)

# Realizar predicciones en el conjunto de validación para los tres modelos.
y_val_pred_lr0 = model_lr0.predict(X_val)
y_val_pred_lr1 = model_lr1.predict(X_val)
y_val_pred_lr2 = model_lr2.predict(X_val)

# Calcular métricas de desempeño para los tres modelos en el conjunto de validación.
accuracy_lr0 = accuracy_score(y_val, y_val_pred_lr0)
accuracy_lr1 = accuracy_score(y_val, y_val_pred_lr1)
accuracy_lr2 = accuracy_score(y_val, y_val_pred_lr2)

# Imprimir las métricas de desempeño antes y después de aplicar regularización.
print("Métricas de desempeño en el conjunto de validación:")
print(f"Antes de regularización (lr0) - Precisión: {accuracy_lr0:.2f}")
print(f"Regularización L1 (lr1) - Precisión: {accuracy_lr1:.2f}")
print(f"Regularización L2 (lr2) - Precisión: {accuracy_lr2:.2f}")

# Visualización de la matriz de confusión para el modelo con regularización L2 (lr2).
conf_matrix_lr2 = confusion_matrix(y_val, y_val_pred_lr2)
sns.heatmap(conf_matrix_lr2, annot=True, cmap='Blues')
plt.xlabel("Predicciones")
plt.ylabel("Valores Reales")
plt.title("Matriz de Confusión (lr2)")
plt.show()

# Calcular métricas de clasificación para los tres modelos en el conjunto de validación.
precision_lr0 = precision_score(y_val, y_val_pred_lr0)
recall_lr0 = recall_score(y_val, y_val_pred_lr0)
f1_lr0 = f1_score(y_val, y_val_pred_lr0)

precision_lr1 = precision_score(y_val, y_val_pred_lr1)
recall_lr1 = recall_score(y_val, y_val_pred_lr1)
f1_lr1 = f1_score(y_val, y_val_pred_lr1)

precision_lr2 = precision_score(y_val, y_val_pred_lr2)
recall_lr2 = recall_score(y_val, y_val_pred_lr2)
f1_lr2 = f1_score(y_val, y_val_pred_lr2)

# Crear una figura para comparar las métricas de F1-score.
plt.figure(figsize=(10, 5))
plt.bar([0, 1, 2], [f1_lr0, f1_lr1, f1_lr2], tick_label=['lr0', 'lr1', 'lr2'], color=['blue', 'green', 'red'])
plt.title("Comparación de F1-score en Validación")
plt.ylabel("F1-score")
plt.savefig("f1_score.png")

# Crear una figura para comparar las métricas de precisión y recall.
plt.figure(figsize=(10, 5))
plt.bar([0, 1, 2], [precision_lr0, precision_lr1, precision_lr2], tick_label=['lr0', 'lr1', 'lr2'], color=['blue', 'green', 'red'], label="Precisión")
plt.bar([0, 1, 2], [recall_lr0, recall_lr1, recall_lr2], tick_label=['lr0', 'lr1', 'lr2'], color=['lightblue', 'lightgreen', 'salmon'], label="Recall")
plt.title("Comparación de Precisión y Recall en Validación")
plt.ylabel("Precisión y Recall")
plt.legend()
plt.savefig("precision_recall.png")

# Visualización de la curva ROC para los tres modelos.
y_val_prob_lr0 = model_lr0.predict_proba(X_val)[:, 1]
fpr_lr0, tpr_lr0, _ = roc_curve(y_val, y_val_prob_lr0)
auc_lr0 = roc_auc_score(y_val, y_val_prob_lr0)

y_val_prob_lr1 = model_lr1.predict_proba(X_val)[:, 1]
fpr_lr1, tpr_lr1, _ = roc_curve(y_val, y_val_prob_lr1)
auc_lr1 = roc_auc_score(y_val, y_val_prob_lr1)

y_val_prob_lr2 = model_lr2.predict_proba(X_val)[:, 1]
fpr_lr2, tpr_lr2, _ = roc_curve(y_val, y_val_prob_lr2)
auc_lr2 = roc_auc_score(y_val, y_val_prob_lr2)

plt.figure(figsize=(10, 5))
plt.plot(fpr_lr0, tpr_lr0, label=f"lr0 (AUC = {auc_lr0:.2f})", color='blue')
plt.plot(fpr_lr1, tpr_lr1, label=f"lr1 (AUC = {auc_lr1:.2f})", color='green')
plt.plot(fpr_lr2, tpr_lr2, label=f"lr2 (AUC = {auc_lr2:.2f})", color='red')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC en Validación")
plt.legend()
plt.savefig("roc.png")

# Gráfico de barras para la distribución de etiquetas reales y predichas en Validación.
plt.figure(figsize=(8, 5))
bar_width = 0.2
index = np.arange(2)
plt.bar(index, [sum(y_val == 0), sum(y_val_pred_lr0 == 0)], bar_width, label="lr0", color='blue')
plt.bar(index + bar_width, [sum(y_val == 1), sum(y_val_pred_lr0 == 1)], bar_width, label="lr0 Predichos", color='lightblue')

plt.bar(index + 2 * bar_width, [sum(y_val == 0), sum(y_val_pred_lr2 == 0)], bar_width, label="lr2", color='red')
plt.bar(index + 3 * bar_width, [sum(y_val == 1), sum(y_val_pred_lr2 == 1)], bar_width, label="lr2 Predichos", color='salmon')

plt.xticks(index + bar_width, ("Clase 0", "Clase 1"))
plt.xlabel("Clases")
plt.ylabel("Cantidad")
plt.title("Distribución de Clases en Validación")
plt.legend()
plt.savefig("distribucion_clases.png")

# Métricas de precisión en el conjunto de validación para los tres modelos.
precisions = [accuracy_lr0, accuracy_lr1, accuracy_lr2]
models = ['lr0', 'lr1', 'lr2']

plt.figure(figsize=(10, 5))
plt.bar(models, precisions, color=['blue', 'green', 'red'])
plt.xlabel("Modelo")
plt.ylabel("Precisión en Validación")
plt.title("Comparación de Precisión en Validación entre Modelos")
plt.ylim(0.7, 0.9)  # Establecer límites en el eje y para una mejor visualización
plt.savefig("precision_validacion.png")
"""