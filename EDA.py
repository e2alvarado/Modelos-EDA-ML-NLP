import pandas as pd
import seaborn as sns
from datetime import datetime
from wordcloud import WordCloud 
from textblob import TextBlob 


# Obtener la fecha actual
fecha_actual = datetime.now().strftime('%d-%m-%Y')

# Ruta del archivo CSV generado por scrape-diario.py con la fecha actual
ruta_csv_scrape_diario = f'Noticias_{fecha_actual}.csv'

# Leer el DataFrame desde el archivo CSV
df_noticias = pd.read_csv(ruta_csv_scrape_diario)




import matplotlib.pyplot as plt
import seaborn as sns

# Frecuencia de noticias por categoría
plt.figure(figsize=(10, 6))
sns.countplot(data=df_noticias, x='Categoría', order=df_noticias['Categoría'].value_counts().index)
plt.title('Frecuencia de Noticias por Categoría')
plt.xlabel('Categoría')
plt.ylabel('Cantidad de Noticias')
plt.xticks(rotation=90)
plt.show()





# Obtener la cantidad de noticias por cada autor
cantidad_por_autor = df_noticias['Autor'].value_counts()

# Visualizar los autores más prolíficos
print(cantidad_por_autor.head(10))

# Graficar la cantidad de noticias por autor (Top 10)
plt.figure(figsize=(20, 6))
cantidad_por_autor.head(10).plot(kind='bar', color='skyblue')
plt.title('Cantidad de Noticias por Autor (Top 10)')
plt.xlabel('Autor')
plt.ylabel('Cantidad de Noticias')
plt.xticks(rotation=45)
plt.show()




# Copia del DataFrame original
df_noticias_copia = df_noticias.copy()

# Convierte la columna 'Fecha' al tipo de datos datetime
df_noticias_copia['Fecha'] = pd.to_datetime(df_noticias_copia['Fecha'], errors='coerce')

# Define una función para extraer el formato de fecha deseado (mes, día y año)
def procesar_fecha(fecha):
    if pd.isnull(fecha):
        return "Fecha no válida"
    mes = fecha.month_name()[:3]
    dia = fecha.day
    año = fecha.year
    return f"{mes} {dia}, {año}"

# Aplica la función a la columna 'Fecha'
df_noticias_copia['Fecha'] = df_noticias_copia['Fecha'].apply(procesar_fecha)

# Muestra el DataFrame resultante
print(df_noticias_copia)





# Convierte la columna 'Fecha' al tipo de datos datetime si no lo has hecho ya
df_noticias['Fecha'] = pd.to_datetime(df_noticias['Fecha'], errors='coerce')

# Ordena el DataFrame por fecha de forma descendente
df_noticias = df_noticias.sort_values(by='Fecha', ascending=False)

# Crea un gráfico de líneas
plt.figure(figsize=(10, 6))
plt.plot(df_noticias['Fecha'], range(1, len(df_noticias) + 1), marker='o', linestyle='-')

# Personaliza el gráfico
plt.title('Cantidad de Noticias desde la Última Fecha hasta la Más Reciente')
plt.xlabel('Fecha')
plt.ylabel('Cantidad de Noticias')
plt.xticks(rotation=45)
plt.grid(True)

# Muestra el gráfico
plt.show()




from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import seaborn as sns
import pandas as pd

# Descargar stopwords de NLTK
nltk.download('stopwords')

# Lista de stopwords de NLTK
stop_words = set(stopwords.words('spanish'))  # Puedes cambiar 'spanish' por el idioma que necesites

# Palabras adicionales a eliminar
additional_stopwords = ['portada', 'dos', 'ser', 'año', 'keywords', 'ser', 'ver', 'tema', 'puede', 'foto:', 'años', 'nuevo', 'tras', 'solo', 'Foto:', 'hacer', 'mejor', 'San', 'san', 'sido', 'así', 'cada']

# Agregar palabras adicionales a la lista de stopwords
stop_words.update(additional_stopwords)

# Obtener las palabras más comunes excluyendo stopwords y palabras monosílabas
todas_palabras = " ".join(df_noticias['Contenido']).split()
palabras_filtradas = [word for word in todas_palabras if word.lower() not in stop_words and len(word) > 2]  # Filtrar palabras con más de 2 caracteres
palabras_comunes_filtradas = Counter(palabras_filtradas).most_common(20)

# Crear un WordCloud de las palabras más comunes filtradas
wordcloud_filtrado = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(palabras_comunes_filtradas))
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud_filtrado, interpolation='bilinear')
plt.axis('off')
plt.title('Palabras Más Comunes en las Noticias (sin stopwords ni palabras específicas)')
plt.show()

# Convertir el resultado de Counter en un DataFrame de pandas para trazar el histograma
df_palabras = pd.DataFrame(palabras_comunes_filtradas, columns=['Palabra', 'Frecuencia'])

# Crear un gráfico de distribución utilizando seaborn
plt.figure(figsize=(10, 6))
sns.histplot(df_palabras['Frecuencia'], kde=True)
plt.title('Distribución de Frecuencia de las Palabras')
plt.xlabel('Frecuencia')
plt.ylabel('Número de Palabras')
plt.show()

# Crear un gráfico de barras para las palabras más comunes
plt.figure(figsize=(10, 6))
sns.barplot(data=df_palabras, x='Palabra', y='Frecuencia', palette='viridis')
plt.title('Palabras Más Comunes en las Noticias')
plt.xlabel('Palabras')
plt.ylabel('Frecuencia')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()




# Función para análisis de sentimientos utilizando TextBlob
def analisis_sentimientos(texto):
    blob = TextBlob(texto)
    return blob.sentiment.polarity  # Obtener la polaridad del sentimiento

# Aplicar análisis de sentimientos al contenido de las noticias
df_noticias['Polaridad'] = df_noticias['Contenido'].apply(analisis_sentimientos)

# Mostrar la polaridad de sentimiento promedio por todas las categorías únicas
polaridad_promedio = df_noticias.groupby('Categoría')['Polaridad'].mean()

# Asegurarse de que todas las categorías estén presentes, incluso si no hay contenido analizable
todas_categorias = df_noticias['Categoría'].unique()
polaridad_promedio = polaridad_promedio.reindex(todas_categorias)

print(polaridad_promedio)





from textblob import TextBlob

# Supongamos que 'Contenido' es la columna que contiene el texto de las noticias en tu DataFrame df_noticias

# Calcular la polaridad para el contenido de las noticias
df_noticias['Polaridad'] = df_noticias['Contenido'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Ahora, puedes calcular la polaridad promedio por categoría
polaridad_promedio = df_noticias.groupby('Categoría')['Polaridad'].mean().reset_index()

# Graficar la polaridad promedio por categoría
plt.figure(figsize=(10, 6))
sns.barplot(data=polaridad_promedio, x='Categoría', y='Polaridad', palette='viridis')
plt.title('Polaridad Promedio por Categoría')
plt.xlabel('Categoría')
plt.ylabel('Polaridad Promedio')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()