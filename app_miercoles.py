import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.title('Modulo 10 - Maestria en Ciencia de Datos')

st.subheader('Cargar Datos')

file=st.file_uploader('Cargar Archivo CSV',type='csv')

# converit a dataframe
df=pd.read_csv(file)

st.write(df)

st.subheader('Informacion del dataframe')

st.write(df.info())

st.subheader('Estadisticas del dataframe')

st.write(df.describe())

# contar el numero de valores nulos

st.subheader('Contar valores nulos')

st.write(df.isnull().sum())

# eliminar valores nulos

st.subheader('Eliminar valores nulos')

df.dropna(inplace=True)
st.write(df.isnull().sum())

# grafico de barras

st.subheader('Grafico de barras')

import plotly.express as px

fig=px.histogram(df,x='sexo')
st.plotly_chart(fig)

# agregar un checkbox para selecciona columnas y graficar

st.subheader('Grafico de barras interactivo')

columnas = df.columns.tolist()

columnas_seleccionadas=st.multiselect('Selecciona Columnas',columnas)

# crear un grafico de correlacion con las columnas seleccionadas en seaborn

import seaborn as sns

st.write(columnas_seleccionadas)

if len(columnas_seleccionadas)>0:
    st.write('Grafico de correlacion')
    fig,ax=plt.subplots()
    ax.scatter(df[columnas_seleccionadas[0]],df[columnas_seleccionadas[1]],color='blue',alpha=0.5)
    st.pyplot(fig)
else:
    st.write('Selecciona una columna')

# agregar la libreria para dividir dataset en entrenamiento y test

from sklearn.model_selection import train_test_split

# vamos a dividir el dataset en entrenamiento y test

X=df.drop(['sexo'],axis=1)
y=df['sexo']

# vamos a agregar un slider para seleccionar el tamaño del test

test_size=st.slider('Tamaño del Test',min_value=0.01,max_value=0.99,value=0.3,step=0.1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size)

st.write('tamaño del entrenamiento',X_train.shape)
st.write('tamaño del test',X_test.shape)