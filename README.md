# Momento de Retroalimentación: Módulo 2 Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución. (Portafolio Implementación)

# Datos de la entrega
Nombre: Pablo González de la Parra

Matrícula: A01745096

Fecha: 11/09/2023

# Descripción del repositorio

De acuerdo con lo establecido en la actividad, el repositorio contiene el código fuente que implemente un algoritmo de aprendizaje máquina en Python utilizando un framework o librería dedicada para su funcionamiento.

En este caso se decidió implementar el algoritmo denominado <b>logistic regression</b> (Regresión logística) de manera que clasifique la información de un dataset en específico.

En el archivo denominado ```logistic_regression.py``` se encuentra la implementación del algoritmo en cuestión, el cual posee tanto el algoritmo como el entrenamiento y validación de este.

El algoritmo funciona con la división del dataset en 80% de entrenamiento y 20% de testing. De igual manera, el 80% de entrenamiento se divide en 20% de validación, el cual permite modificar los hiperparámetros para encontrar la solución más optima.

La solución es impresa a la terminal, la cual consiste de dos partes:
1. Métricas de evaluación del modelo en el set de validación.
    - Puntuación de precisión
    - Puntuación de recall
    - Puntuación de F1
2. Matriz de confusión del modelo en el set de testing.

Ejemplo:
```
Precisión: 0.81
Recall: 0.72
F1-score: 0.76
Matriz de Confusión:
[[415  60]
 [ 98 253]]
```

# Manual de ejecución
Para poder ejecutar el algoritmo se debe de tener instalado Python 3.8 o superior, y seguir los siguienets pasos:

1. Clonar el repositorio en la computadora local.
2. Abrir una terminal en la carpeta del repositorio.
4. Instalar las librerias y dependenicas utilizando <b>pip</b> y los siguientes comandos.

```
pip install sklearn

pip install scikit-learn

pip install matplotlib

pip install seaborn
```

3. Ejecutar el comando ```python logistic_regression.py```

# Referencias
Origen del dataset utilizado:

### California Housing Prices (scikit-learn)
Este dataset contiene información sobre el censo de California de 1990. El dataset contiene 20,640 registros y 10 atributos. El objetivo es predecir el precio medio de las casas en un distrito (llamado blockgroup) en base a los atributos del distrito.