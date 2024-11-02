import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.datasets import load_iris

#cargar el modelo previamente entrendo
model = load_model('iris_model.h5')

#cargar modelo
iris = load_iris()
class_names = iris.target_names

#configuracion de la aplicacion stremlit
st.title('Clasificacion de flores Iris')
st.write('Esta aplicacion predice la clase de una flor Iris.')

#Seleccion de la especie de flor
sepal_length = st.slider('Longitud de sepalo', 4.0, 8.0, 5.0)
sepal_width = st.slider('Ancho de sepalo', 2.0, 4.5, 3.0)
petal_length = st.slider('Longitud de petalo', 1.0, 7.0, 1.5)
petal_width = st.slider('Ancho de petalo', 0.1, 2.5, 0.2)
#boton para predecir
if st.button('Predecir'):
    #crear un arreglo con los datos de la flor
    X = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    # predecir la clase de la flor
    prediction = model.predict(X)
    prediction_class=class_names[np.argmax(prediction)]
    st.write(f'La especie de la flor es: {prediction_class}')
