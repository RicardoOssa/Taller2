import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

# Cargar el conjunto de datos Iris
iris = load_iris()
x = iris.data
y = iris.target
# convertir las etiquetas a codificacion
y = to_categorical(y)
# dividir el conjunto de datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#inicializar el modelo secuencial
model = Sequential()
# capa de entrada
model.add(Dense(8, input_dim=4, activation='relu'))
# capa oculta
model.add(Dense(6, activation='relu'))
# capa de salida
model.add(Dense(3, activation='softmax'))
# compilar modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# entrenar el modelo
model.fit(x_train, y_train, epochs=100, batch_size=4, validation_data=(x_test, y_test))
#Guardar modelo
model.save('iris_model.h5')
print('Modelo guardado como iris model.h5')