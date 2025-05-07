import matplotlib

matplotlib.use('Agg')  # Establecer el backend a 'Agg' para evitar problemas con GUI en servidores

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


# Función para generar recomendaciones agrícolas basadas en las condiciones climáticas
def generar_recomendaciones_texto(temperaturas_futuras, precipitacion_media, humedad_media):
    # Convertir las predicciones a flotantes estándar de Python (para evitar np.float64)
    temperaturas_futuras = [float(temp) for temp in temperaturas_futuras]

    # Recomendaciones basadas en las condiciones climáticas proyectadas (simplificado)
    texto_entrada = f"Las predicciones de temperatura para los próximos 8 años son las siguientes: {temperaturas_futuras}. "

    if max(temperaturas_futuras) > 30:
        texto_entrada += "Con temperaturas muy altas proyectadas, se recomienda optar por cultivos más resistentes al calor, como el maíz, sorgo o ciertos tipos de frijoles. "
        texto_entrada += "También es importante implementar técnicas de riego por goteo y usar sistemas de sombreo para reducir el estrés térmico."
    elif precipitacion_media > 150:
        texto_entrada += "Con una alta probabilidad de lluvias, es recomendable cultivar arroz o vegetales que toleren el exceso de agua. "
        texto_entrada += "Es esencial implementar técnicas de drenaje para evitar el anegamiento de los cultivos."
    else:
        texto_entrada += "Con temperaturas moderadas y lluvias regulares, se sugiere diversificar los cultivos. Se pueden aplicar prácticas agroforestales para proteger los suelos y mejorar la calidad del agua."

    return texto_entrada


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
            año = data.get("año")

            if not año:
                return JsonResponse({"error": "Falta el campo de año"}, status=400)

            # Usar pandas para asegurarnos de que los datos tienen nombres de columnas correctos
            import pandas as pd
            input_data = pd.DataFrame([[temperatura, precipitacion, humedad, 0, año]],
                                      columns=["temperatura", "precipitacion_mm", "humedad_relativa",
                                               "velocidad_viento", "year"])

            if model is None:
                return JsonResponse({"error": f"Error al cargar el modelo: {model_error}"}, status=500)

            recomendaciones = []
            points = []  # Lista para los puntos del mapa de calor

            # Definir el rango de años para las recomendaciones
            años_futuros = list(range(año, año + 8))  # Recomendaciones para los próximos 8 años

            temperaturas_futuras = []
            for year in años_futuros:
                input_data = pd.DataFrame([[temperatura, precipitacion, humedad, 0, year]],
                                          columns=["temperatura", "precipitacion_mm", "humedad_relativa",
                                                   "velocidad_viento", "year"])
                recomendacion = model.predict(input_data)
                temperaturas_futuras.append(recomendacion[0])  # Almacenamos la recomendación para cada año

                # Añadir más puntos para cada recomendación (más dispersión)
                for _ in range(10):  # Generamos 10 puntos adicionales por recomendación
                    lat_offset = random.uniform(-0.1, 0.1)  # Variación aleatoria mayor en latitud
                    lon_offset = random.uniform(-0.1, 0.1)  # Variación aleatoria mayor en longitud

                    # Añadir los puntos al mapa (usamos latitud, longitud y la recomendación como "valor" de calor)
                    points.append([lat + lat_offset, lon + lon_offset, recomendacion[0]])

            # Crear fluctuaciones para hacer el gráfico dinámico
            medias_recomendaciones = []
            for i, temp in enumerate(temperaturas_futuras):
                # Añadir fluctuaciones para cada predicción
                fluctuation = random.uniform(-0.5, 0.5)  # Variación aleatoria de temperatura para cada año
                fluctuated_temp = temp + fluctuation
                medias_recomendaciones.append(fluctuated_temp)

            # Crear un mapa de calor con folium
            m = folium.Map(location=[lat, lon], zoom_start=10)

            # Crear los puntos del mapa de calor (coordenadas y valores de recomendación)
            HeatMap(points).add_to(m)

            # Guardar el mapa como archivo HTML en memoria
            map_file = io.BytesIO()
            m.save(map_file, close_file=False)

            # Convertir el archivo HTML a base64
            map_base64 = base64.b64encode(map_file.getvalue()).decode('utf-8')

            # Crear el gráfico de recomendación de temperatura a lo largo de los años
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(años_futuros, medias_recomendaciones, marker='o', color='r', label='Recomendación de Temperatura')
            ax.set_xlabel('Año')
            ax.set_ylabel('Temperatura (°C)')
            ax.set_title('Evolución de la Temperatura Recomendada')
            ax.grid(True)
            ax.legend()

            # Convertir el gráfico a imagen base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')

            # Generar recomendaciones de texto utilizando los valores de temperatura (fluctuados para cada año)
            texto_recomendaciones = generar_recomendaciones_texto(medias_recomendaciones, precipitacion, humedad)

            return JsonResponse({
                'recomendaciones': medias_recomendaciones,
                'grafico': map_base64,  # Ahora devolvemos el mapa de calor en base64
                'grafico_linea': img_str,  # Devolvemos el gráfico de la progresión de temperaturas en base64
                'coordenadas': coordenadas[departamento],
                'analisis_texto': texto_recomendaciones  # Agregar las recomendaciones generadas
            })

        except Exception as e:
            print(f"Error en el servidor: {str(e)}")  # Depuración de errores
            return JsonResponse({"error": f"Hubo un error en el servidor: {str(e)}"}, status=500)
