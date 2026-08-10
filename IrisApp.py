import streamlit as st
import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.write(''' # 🌸 Clasificador de Iris ''')

def user_input_features():
    petal_length = st.number_input('Longitud del pétalo (cm):', min_value=1.0, max_value=7.0, value=3.0, step=0.1)
    petal_width = st.number_input('Ancho del pétalo (cm):', min_value=0.1, max_value=2.5, value=1.0, step=0.1)
    sepal_length = st.number_input('Longitud del sépalo (cm):', min_value=4.0, max_value=8.0, value=5.0, step=0.1)
    sepal_width = st.number_input('Ancho del sépalo (cm):', min_value=2.0, max_value=4.5, value=3.0, step=0.1)

   return pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                       columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

df = user_input_features()

# Verificar si existe el CSV
if not os.path.exists('iris_dataset.csv'):
    st.error('❌ No se encuentra el archivo iris_dataset.csv')
    st.stop()

# Cargar dataset desde CSV
iris_df = pd.read_csv('iris_dataset.csv')

# Separar características y target
X = iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = iris_df['species']

# Codificar especies
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Modelo
model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predicción
prediction = model.predict(df)[0]
species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

st.subheader('🎯 Predicción')
st.write(f'**La flor es:** {species[prediction]}')

# Mostrar probabilidades
prediction_proba = model.predict_proba(df)
st.subheader('📊 Probabilidades')
prob_df = pd.DataFrame({
    'Especie': ['Setosa', 'Versicolor', 'Virginica'],
    'Probabilidad': prediction_proba[0]
})

st.bar_chart(prob_df.set_index('Especie'))

print("\n✅ Archivo IrisApp.py creado en Colab")
