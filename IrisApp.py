import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.write(''' # Clasificador de Iris ''')
st.image("iris.webp", caption="Tipos de lor Iris.")

def user_input_features():
    petal_length = st.number_input('Longitud del pétalo (cm):', min_value=1, max_value=7, value=3, step=1)
    petal_width = st.number_input('Ancho del pétalo (cm):', min_value=0, max_value=2, value=1, step=1)
    sepal_length = st.number_input('Longitud del sépalo (cm):', min_value=4, max_value=8, value=5, step=1)
    sepal_width = st.number_input('Ancho del sépalo (cm):', min_value=2, max_value=4, value=3, step=1)
    
    return pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                       columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

df_input = user_input_features()

# Cargar datos Iris (igual que tu código)
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

# Modelo
model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predicción
prediction = model.predict(df_input)[0]
species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

st.subheader('Predicción')
st.write(f'**La flor es:** {species[prediction]}')

