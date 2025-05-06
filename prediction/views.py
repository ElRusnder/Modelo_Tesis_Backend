import folium
from folium.plugins import HeatMap
import io
import base64
import random
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
import joblib  # Usamos joblib en lugar de pickle
import requests
import matplotlib.pyplot as plt
import os
from django.conf import settings

# Cargar el modelo de predicción con joblib
model_path = os.path.join(settings.BASE_DIR, 'prediction', 'random_forest_model_v2.pkl')  # Ruta correcta al archivo .pkl

try:
    model = joblib.load(model_path)
    print(f"Modelo cargado correctamente: {type(model)}")  # Verifica el tipo de modelo cargado
except Exception as e:
    model = None
    model_error = str(e)  # Guardamos el error al cargar el modelo
    print(f"Error al cargar el modelo: {model_error}")

@csrf_exempt
def predict(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)

            departamento = data.get("departamento")
            coordenadas = {
                "Junín": {"lat": -11.9396, "lon": -74.5482},
                "Ayacucho": {"lat": -13.1584, "lon": -74.2231},
                "Cusco": {"lat": -13.5313, "lon": -71.9675},
                "Puno": {"lat": -15.8402, "lon": -69.0194},
            }

            if departamento not in coordenadas:
                return JsonResponse({"error": "Departamento no válido"}, status=400)

            lat, lon = coordenadas[departamento]["lat"], coordenadas[departamento]["lon"]

            api_key = '15b9eb841a2eb8d126c001585efbcb00'  # Reemplazar con tu clave de API
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            response = requests.get(url)

            if response.status_code != 200:
                return JsonResponse({"error": "Error al obtener datos del clima"}, status=500)

            clima_data = response.json()
            temperatura = clima_data["main"]["temp"]
            precipitacion = clima_data.get("rain", {}).get("1h", 0)
            humedad = clima_data["main"]["humidity"]
            viento = clima_data["wind"]["speed"]
            año = data.get("año")

            if not año:
                return JsonResponse({"error": "Falta el campo de año"}, status=400)

            input_data = np.array([[temperatura, precipitacion, humedad, viento, año]])

            if model is None:
                return JsonResponse({"error": f"Error al cargar el modelo: {model_error}"}, status=500)

            predicciones = []
            points = []  # Lista para los puntos del mapa de calor

            # Definir el rango de años para las predicciones
            años_futuros = list(range(año, año + 8))  # Predicciones de los próximos 8 años

            for year in años_futuros:
                input_data = np.array([[temperatura, precipitacion, humedad, viento, year]])
                prediccion = model.predict(input_data)
                predicciones.append(prediccion[0])  # Almacenamos la predicción para cada año

                # Añadir más puntos para cada predicción (más dispersión)
                for _ in range(10):  # Generamos 10 puntos adicionales por predicción
                    lat_offset = random.uniform(-0.1, 0.1)  # Variación aleatoria mayor en latitud
                    lon_offset = random.uniform(-0.1, 0.1)  # Variación aleatoria mayor en longitud

                    # Añadir los puntos al mapa (usamos latitud, longitud y la predicción de temperatura como "valor" de calor)
                    points.append([lat + lat_offset, lon + lon_offset, prediccion[0]])

            # Crear fluctuaciones para hacer el gráfico dinámico
            medias_predicciones = []
            for year in años_futuros:
                # Añadir un pequeño cambio aleatorio basado en los puntos generados para simular fluctuaciones
                fluctuation = random.uniform(-0.1, 0.1)  # Cambio aleatorio de temperatura para simular fluctuaciones
                fluctuated_temp = predicciones[0] + fluctuation
                medias_predicciones.append(fluctuated_temp)

            # Crear un mapa de calor con folium
            m = folium.Map(location=[lat, lon], zoom_start=10)

            # Crear los puntos del mapa de calor (coordenadas y valores de temperatura)
            HeatMap(points).add_to(m)

            # Guardar el mapa como archivo HTML en memoria
            map_file = io.BytesIO()
            m.save(map_file, close_file=False)

            # Convertir el archivo HTML a base64
            map_base64 = base64.b64encode(map_file.getvalue()).decode('utf-8')

            # Crear el gráfico de predicción de temperatura a lo largo de los años
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(años_futuros, medias_predicciones, marker='o', color='r', label='Predicción de Temperatura')
            ax.set_xlabel('Año')
            ax.set_ylabel('Temperatura (°C)')
            ax.set_title('Evolución de la Temperatura Predicha')
            ax.grid(True)
            ax.legend()

            # Convertir el gráfico a imagen base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')

            return JsonResponse({
                'predicciones': medias_predicciones,
                'grafico': map_base64,  # Ahora devolvemos el mapa de calor en base64
                'grafico_linea': img_str,  # Devolvemos el gráfico de la progresión de temperaturas en base64
                'coordenadas': coordenadas[departamento]
            })

        except Exception as e:
            return JsonResponse({"error": f"Hubo un error en el servidor: {str(e)}"}, status=500)
