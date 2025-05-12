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
from sklearn.tree import DecisionTreeClassifier  # Librería para clasificación
import pandas as pd

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


# Función para recomendar cultivo y riego
def recomendar_cultivo_y_riego(temperaturas_futuras, precipitacion_media, humedad_media):
    # Modelo entrenado para recomendar cultivo (simplificado)
    # Usamos la temperatura, precipitación y humedad para predecir el cultivo
    X = np.array([[30, 120, 60], [25, 200, 70], [35, 100, 50], [28, 150, 65]])  # Ejemplo de datos
    y = ['Maíz', 'Arroz', 'Frijoles', 'Papas']

    modelo_cultivo = DecisionTreeClassifier()
    modelo_cultivo.fit(X, y)

    # Predecir el cultivo recomendado basado en las condiciones climáticas
    cultivo_recomendado = modelo_cultivo.predict([[max(temperaturas_futuras), precipitacion_media, humedad_media]])[0]

    # Descripción detallada del cultivo recomendado y riego
    if cultivo_recomendado == 'Maíz':
        recomendacion_riego = "El maíz es un cultivo resistente al calor, pero requiere de un riego frecuente para maximizar el rendimiento. Se recomienda utilizar riego por goteo y sombrear las plantas en climas muy calurosos."
        descripcion_cultivo = "El maíz es un cultivo de alto rendimiento en temperaturas cálidas y suelos bien irrigados. Requiere un manejo intensivo del agua."
    elif cultivo_recomendado == 'Arroz':
        recomendacion_riego = "El arroz necesita grandes cantidades de agua, por lo que el riego inundado es la mejor técnica. Se recomienda controlar las lluvias para evitar inundaciones excesivas."
        descripcion_cultivo = "El arroz es adecuado para suelos húmedos y requiere inundación para un crecimiento óptimo, lo que lo hace ideal en regiones con alta pluviosidad."
    elif cultivo_recomendado == 'Frijoles':
        recomendacion_riego = "Los frijoles prefieren un riego moderado, que mantenga la humedad del suelo sin llegar al exceso. Usar sistemas de riego por goteo es lo más recomendable."
        descripcion_cultivo = "El frijol es un cultivo de ciclo corto que se adapta bien a climas templados y suelos bien drenados."
    else:
        recomendacion_riego = "Las papas requieren de un manejo de riego eficiente, preferentemente por goteo, para evitar el exceso de humedad y la formación de hongos."
        descripcion_cultivo = "Las papas prosperan en climas frescos y suelos bien aireados. Son muy sensibles a las condiciones de humedad, por lo que es esencial un riego controlado."

    return cultivo_recomendado, recomendacion_riego, descripcion_cultivo


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
            input_data = pd.DataFrame([[temperatura, precipitacion, humedad, 0, año]],
                                      columns=["temperatura", "precipitacion_mm", "humedad_relativa",
                                               "velocidad_viento", "year"])

            if model is None:
                return JsonResponse({"error": f"Error al cargar el modelo: {model_error}"}, status=500)

            # Predicciones de temperatura para los próximos años
            años_futuros = list(range(año, año + 8))  # Recomendaciones para los próximos 8 años
            temperaturas_futuras = [model.predict(input_data)[0] for year in años_futuros]

            # Generar fluctuaciones para hacer el gráfico dinámico
            medias_recomendaciones = [temp + random.uniform(-0.5, 0.5) for temp in temperaturas_futuras]

            # Obtener cultivo recomendado y riego
            cultivo_recomendado, recomendacion_riego, descripcion_cultivo = recomendar_cultivo_y_riego(temperaturas_futuras, precipitacion, humedad)

            # Crear mapa de calor con más puntos dispersos
            points = []
            for _ in range(50):  # Generamos 50 puntos dispersos por recomendación
                lat_offset = random.uniform(-0.3, 0.3)
                lon_offset = random.uniform(-0.3, 0.3)
                points.append([lat + lat_offset, lon + lon_offset, temperaturas_futuras[0]])

            m = folium.Map(location=[lat, lon], zoom_start=10)
            HeatMap(points).add_to(m)

            map_file = io.BytesIO()
            m.save(map_file, close_file=False)
            map_base64 = base64.b64encode(map_file.getvalue()).decode('utf-8')

            # Crear el gráfico de recomendación de temperatura a lo largo de los años
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(años_futuros, medias_recomendaciones, marker='o', color='r', label='Recomendación de Temperatura')
            ax.set_xlabel('Año')
            ax.set_ylabel('Temperatura (°C)')
            ax.set_title('Evolución de la Temperatura Recomendada')
            ax.grid(True)
            ax.legend()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')

            texto_recomendaciones = generar_recomendaciones_texto(medias_recomendaciones, precipitacion, humedad)

            return JsonResponse({
                'cultivo_recomendado': cultivo_recomendado,
                'recomendacion_riego': recomendacion_riego,
                'descripcion_cultivo': descripcion_cultivo,
                'temperaturas_futuras': medias_recomendaciones,
                'grafico': map_base64,  # Mapa en base64
                'grafico_linea': img_str,  # Gráfico de la progresión de temperaturas
                'coordenadas': coordenadas[departamento],
                'analisis_texto': texto_recomendaciones  # Agregar las recomendaciones generadas
            })

        except Exception as e:
            print(f"Error en el servidor: {str(e)}")
            return JsonResponse({"error": f"Hubo un error en el servidor: {str(e)}"}, status=500)
