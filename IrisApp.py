import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.write(''' # Clasificador de Iris ''')
st.image("iris.webp", caption="Especies de la flor Iris.")

def user_input_features():
    petal_length = st.slider('Longitud del pétalo:', 1, 7, 3)
    petal_width = st.slider('Ancho del pétalo:', 0, 2, 1)
    sepal_length = st.slider('Longitud del sépalo:', 4, 8, 5)
    sepal_width = st.slider('Ancho del sépalo:', 2, 4, 3)
    
    return pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                       columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

df_input = user_input_features()

# Cargar datos
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

X = df[iris.feature_names]
y = df['species']

# Codificar y dividir
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Modelo balanceado
model = RandomForestClassifier(n_estimators=150, max_depth=4, max_features='sqrt', random_state=42)
model.fit(X_train, y_train)

# Predicción
prediction = model.predict(df_input)[0]
species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

st.subheader('Predicción')
st.write(f'**La especie de flor iris es:  {species[prediction]} **')
