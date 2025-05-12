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
    temperaturas_futuras = [float(temp) for temp in temperaturas_futuras]
    temperatura_media = sum(temperaturas_futuras) / len(temperaturas_futuras)

    texto_entrada = f"Las predicciones de temperatura para los próximos años son las siguientes: {temperaturas_futuras}. "
    texto_entrada += f"La temperatura media para el rango seleccionado es: {temperatura_media:.2f}°C. "

    if temperatura_media > 30:
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
    temperatura_media = sum(temperaturas_futuras) / len(temperaturas_futuras)  # Calcular la temperatura media

    # Determinar la recomendación en función de la temperatura media y las condiciones climáticas
    if temperatura_media > 30:
        cultivo_recomendado = 'Maíz'
        recomendacion_riego = "El maíz es un cultivo resistente al calor, pero requiere de un riego frecuente para maximizar el rendimiento. Se recomienda utilizar riego por goteo y sombrear las plantas en climas muy calurosos."
        descripcion_cultivo = "El maíz es un cultivo de alto rendimiento en temperaturas cálidas y suelos bien irrigados. Requiere un manejo intensivo del agua."
    elif precipitacion_media > 150:
        cultivo_recomendado = 'Arroz'
        recomendacion_riego = "El arroz necesita grandes cantidades de agua, por lo que el riego inundado es la mejor técnica. Se recomienda controlar las lluvias para evitar inundaciones excesivas."
        descripcion_cultivo = "El arroz es adecuado para suelos húmedos y requiere inundación para un crecimiento óptimo, lo que lo hace ideal en regiones con alta pluviosidad."
    elif temperatura_media < 25:
        cultivo_recomendado = 'Frijoles'
        recomendacion_riego = "Los frijoles prefieren un riego moderado, que mantenga la humedad del suelo sin llegar al exceso. Usar sistemas de riego por goteo es lo más recomendable."
        descripcion_cultivo = "El frijol es un cultivo de ciclo corto que se adapta bien a climas templados y suelos bien drenados."
    else:
        cultivo_recomendado = 'Papas'
        recomendacion_riego = "Las papas requieren de un manejo de riego eficiente, preferentemente por goteo, para evitar el exceso de humedad y la formación de hongos."
        descripcion_cultivo = "Las papas prosperan en climas frescos y suelos bien aireados. Son muy sensibles a las condiciones de humedad, por lo que es esencial un riego controlado."

    # Devolver la recomendación única
    return {
        "cultivo_recomendado": cultivo_recomendado,
        "recomendacion_riego": recomendacion_riego,
        "descripcion_cultivo": descripcion_cultivo
    }


@csrf_exempt
def predict(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)

            # Obtener la fecha de inicio y final de la solicitud
            start_year = int(data.get("start_year"))
            start_month = int(data.get("start_month"))
            end_year = int(data.get("end_year"))
            end_month = int(data.get("end_month"))

            # Validaciones de las fechas
            if end_year - start_year > 8:
                return JsonResponse({"error": "El rango de predicción no puede ser mayor a 8 años."}, status=400)

            if end_year == start_year and end_month < start_month:
                return JsonResponse({"error": "El mes final no puede ser anterior al mes de inicio."}, status=400)

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

            # Generar las predicciones para los meses y años seleccionados
            años_futuros = list(range(start_year, end_year + 1))
            temperaturas_futuras = []
            meses_futuros = []
            for year in años_futuros:
                for month in range(1, 13):
                    if year == start_year and month < start_month:
                        continue
                    if year == end_year and month > end_month:
                        continue
                    input_data = pd.DataFrame([[temperatura, precipitacion, humedad, 0, year]],
                                              columns=['humedad_relativa', 'precipitacion_mm', 'velocidad_viento', 'año', 'mes'])
                    temperaturas_futuras.append(model.predict(input_data)[0])
                    meses_futuros.append(f'{month}-{year}')

            # Crear fluctuaciones para el gráfico
            medias_recomendaciones = [temp + random.uniform(-0.5, 0.5) for temp in temperaturas_futuras]

            # Obtener las recomendaciones de cultivo y riego
            recomendaciones = recomendar_cultivo_y_riego(temperaturas_futuras, precipitacion, humedad)

            # Crear mapa de calor
            points = []
            for _ in range(50):
                lat_offset = random.uniform(-0.3, 0.3)
                lon_offset = random.uniform(-0.3, 0.3)
                points.append([lat + lat_offset, lon + lon_offset, temperaturas_futuras[0]])

            m = folium.Map(location=[lat, lon], zoom_start=10)
            HeatMap(points).add_to(m)

            map_file = io.BytesIO()
            m.save(map_file, close_file=False)
            map_base64 = base64.b64encode(map_file.getvalue()).decode('utf-8')

            # Crear gráfico de barras para la temperatura
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(meses_futuros, medias_recomendaciones, color='skyblue')
            ax.set_xlabel('Mes-Año')
            ax.set_ylabel('Temperatura (°C)')
            ax.set_title('Predicción de Temperatura por Mes y Año')
            ax.set_xticklabels(meses_futuros, rotation=45, ha='right')
            ax.grid(True)

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')

            texto_recomendaciones = generar_recomendaciones_texto(medias_recomendaciones, precipitacion, humedad)

            return JsonResponse({
                'cultivo_recomendado': recomendaciones,  # Se pasan las recomendaciones completas
                'grafico': map_base64,  # Mapa en base64
                'grafico_barras': img_str,  # Gráfico de barras de temperatura
                'coordenadas': coordenadas[departamento],
                'analisis_texto': texto_recomendaciones  # Recomendaciones generadas
            })

        except Exception as e:
            print(f"Error en el servidor: {str(e)}")
            return JsonResponse({"error": f"Hubo un error en el servidor: {str(e)}"}, status=500)
