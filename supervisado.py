import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

num_muestras = 100
rutas = ['R' + str(i) for i in np.random.randint(1, 21, num_muestras)] 
duracion_viaje = np.random.randint(10, 60, num_muestras)  
num_pasajeros = np.random.randint(10, 100, num_muestras)  

data = pd.DataFrame({
    'Ruta': rutas,
    'Duración_Viaje': duracion_viaje,
    'Num_Pasajeros': num_pasajeros
})

# Convierte los dats de txto en numericos para ser procesados por pandas
data_dummies = pd.get_dummies(data, columns=['Ruta'])

X = data_dummies.drop(columns='Duración_Viaje')
y = data['Duración_Viaje']

#entrenamento y prueba.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar el modelo de regresión lineal
modelo = LinearRegression()

# Entrenar el modelo
modelo.fit(X_train, y_train)

#hacieendo predicciones
predicciones = modelo.predict(X_test)

# Evaluar el modelo
error_medio = mean_squared_error(y_test, predicciones)
ajuste_modelo = r2_score(y_test, predicciones)

print("Resultados del Modelo de Regresión Lineal:")
print(f"Error cuadrático medio (MSE): {error_medio:.2f}")
print(f"Coeficiente de determinación (R^2): {ajuste_modelo:.2f}")

# Mostrar algunas predicciones comparadas con los valores reales
comparacion = pd.DataFrame({
    'Real': y_test,
    'Predicción': predicciones
})
print("\nComparación de valores reales y predicciones:")
print(comparacion.head())
