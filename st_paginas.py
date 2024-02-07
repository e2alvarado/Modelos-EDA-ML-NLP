import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
from datetime import datetime
from wordcloud import WordCloud 
from textblob import TextBlob 
import plotly.express as px
from collections import Counter
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import spacy
import string
from scipy.cluster.hierarchy import dendrogram, linkage
from NLP_M1 import cargar_modelo, procesar_noticia_seleccionada
import pip
pip.main(["install", "openpyxl"])

st.set_option('deprecation.showPyplotGlobalUse', False)

# Cargar datos
fecha_actual = datetime.now().strftime('%d-%m-%Y')
ruta_csv_scrape_diario = f'Noticias_{fecha_actual}.csv'
df_noticias = pd.read_csv(ruta_csv_scrape_diario)

# Cargar el modelo de spaCy
modelo_spacy = cargar_modelo()


# Cargar las variables necesarias para el modelo de Machine Learning
with open('variables_modelo.pkl', 'rb') as f:
    variables_modelo = pickle.load(f)

# Extraer las variables
X_train = variables_modelo['X_train']
X_test = variables_modelo['X_test']
y_train = variables_modelo['y_train']
y_test = variables_modelo['y_test']
vectorizer = variables_modelo['vectorizer']
selector = variables_modelo['selector']
svm_model = variables_modelo['svm_model']

# Utilizar todo el conjunto de prueba para la evaluación
X_test_sampled = X_test
X_test_tfidf = vectorizer.transform(X_test_sampled)
X_test_selected = selector.transform(X_test_tfidf)

# Realizar predicciones con el modelo SVM
y_pred = svm_model.predict(X_test_selected)

# Número de noticias recolectadas
num_noticias = len(df_noticias)

# Título de la aplicación
st.title('Extracción y Análisis de Noticias de El Salvador 📰')
st.write(
    "En este proyecto, se realizó la extracción automatizada de noticias desde la página web salvadoreña **El Salvador** "
    "(https://www.elsalvador.com/). Utilizando técnicas de web scraping con Python, se recopilaron datos de diversas secciones "
    "del sitio, como noticias locales, internacionales y deportes."
)

# Barra lateral para seleccionar la página
pagina_seleccionada = st.sidebar.selectbox("Seleccionar Página", ["Análisis Exploratorio de Datos (EDA)", "Modelo SVM", "Modelos NLP"])

# Página 1: Análisis Exploratorio de Datos (EDA)
if pagina_seleccionada == "Análisis Exploratorio de Datos (EDA)":
    st.markdown("## Análisis Exploratorio de Datos (EDA)")

# Resto del código para la página de EDA
    st.write('Tabla de Noticias: Esta tabla presenta noticias recopiladas de diferentes secciones del periódico digital El Salvador. Se utilizaron técnicas de web scraping para extraer enlaces de noticias de diversas categorías y librerías tales como: requests, BeautifulSoup, pandas, datatime, urllib.parse.', df_noticias)

    def visualizar_frecuencia_por_categoria():
        fig = px.bar(df_noticias, x='Categoría', title='Frecuencia de Noticias por Categoría')
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig)

    st.write(f"Se han recopilado {num_noticias} noticias.")

# Función para visualizar la frecuencia de noticias por categoría
    def visualizar_frecuencia_por_categoria():
        st.subheader('Frecuencia de Noticias por Categoría')
        fig = px.bar(df_noticias, x='Categoría' )
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig)

# Función para visualizar la cantidad de noticias por autor (Top 10)
    def visualizar_cantidad_por_autor():
        st.subheader('TOP 10 de Autores con más noticias')
        cantidad_por_autor = df_noticias['Autor'].value_counts().head(10).reset_index()
        cantidad_por_autor.columns = ['Autor', 'Cantidad']
        fig = px.bar(cantidad_por_autor, x='Autor', y='Cantidad')
        st.plotly_chart(fig)

# Función para visualizar el gráfico de líneas de la cantidad de noticias desde la última fecha hasta la más reciente
    def visualizar_grafico_lineas(df_noticias):
        st.subheader('Cantidad de noticias por mes')
        df_noticias['Fecha'] = pd.to_datetime(df_noticias['Fecha'], errors='coerce')
        df_noticias = df_noticias.sort_values(by='Fecha', ascending=False)
    
# Establecer el rango del eje y de 0 a 200
        fig = px.line(df_noticias, x='Fecha', y=range(1, len(df_noticias) + 1))
        fig.update_layout(xaxis_title='Fecha', yaxis_title='Cantidad de Noticias')
        fig.update_yaxes(range=[0, 1350])  # Establecer el rango del eje y
        st.plotly_chart(fig)


# Función para visualizar el WordCloud de las palabras más comunes en las noticias
    def visualizar_wordcloud():
        st.subheader('Palabras más comunes en las noticias')
        todas_palabras = " ".join(df_noticias['Contenido']).split()
        stop_words = set(stopwords.words('spanish'))
        additional_stopwords = ['portada', 'dos', 'ser', 'año', 'keywords', 'ser', 'ver', 'tema', 'puede', 'foto:', 'años', 'nuevo', 'tras', 'solo', 'Foto:', 'hacer', 'mejor', 'San', 'san', 'sido', 'así', 'cada']
        stop_words.update(additional_stopwords)
        palabras_filtradas = [word for word in todas_palabras if word.lower() not in stop_words and len(word) > 2]
        palabras_comunes_filtradas = Counter(palabras_filtradas).most_common(20)
        wordcloud_filtrado = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(palabras_comunes_filtradas))
        fig = px.imshow(wordcloud_filtrado)
        fig.update_layout(title='WordCloud de las Palabras Más Comunes en las Noticias')
        st.plotly_chart(fig)

# Función para visualizar el análisis de sentimientos promedio por categoría
    def visualizar_analisis_sentimientos():
        st.subheader('Polaridad Promedio por Categoría')
        df_noticias['Polaridad'] = df_noticias['Contenido'].apply(lambda x: TextBlob(x).sentiment.polarity)
        polaridad_promedio = df_noticias.groupby('Categoría')['Polaridad'].mean().reset_index()
        fig = px.bar(polaridad_promedio, x='Categoría', y='Polaridad', color='Polaridad', color_continuous_scale='viridis')
        st.plotly_chart(fig)

# Función para visualizar la distribución de frecuencia de las palabras
    def visualizar_distribucion_frecuencia_palabras():
        st.subheader('Distribución de Frecuencia de las Palabras')
        stop_words = set(stopwords.words('spanish'))
        additional_stopwords = ['portada', 'dos', 'ser', 'año', 'keywords', 'ser', 'ver', 'tema', 'puede', 'foto:', 'años', 'nuevo', 'tras', 'solo', 'Foto:', 'hacer', 'mejor', 'San', 'san', 'sido', 'así', 'cada']
        stop_words.update(additional_stopwords)
        todas_palabras = " ".join(df_noticias['Contenido']).split()
        palabras_filtradas = [word for word in todas_palabras if word.lower() not in stop_words and len(word) > 2]
        palabras_comunes_filtradas = Counter(palabras_filtradas).most_common(20)
    
# Crear un DataFrame a partir de las palabras comunes filtradas
        df_palabras_comunes = pd.DataFrame(palabras_comunes_filtradas, columns=['Palabra', 'Frecuencia'])
    
# Crear gráfico interactivo con Plotly Express
        fig = px.bar(df_palabras_comunes, x='Palabra', y='Frecuencia')
        st.plotly_chart(fig)

# Visualizar gráficas de EDA
    visualizar_frecuencia_por_categoria()
    visualizar_cantidad_por_autor()
    visualizar_grafico_lineas(df_noticias)
    visualizar_wordcloud()
    visualizar_analisis_sentimientos()
    visualizar_distribucion_frecuencia_palabras()


# Página 2: Machine Learning
elif pagina_seleccionada == "Modelo SVM":

    # Sección de Machine Learning centrada
    st.markdown("<h1 style='text-align: center;'>Sección de Machine Learning</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Modelo SVM</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Poniendo a prueba los resultados del modelo Machine Learning</p>", unsafe_allow_html=True)

    # Visualizar las predicciones
    st.subheader('Resultados de las Predicciones')
    resultados_df = pd.DataFrame({'Texto': X_test_sampled, 'Predicción': y_pred})
    st.write(resultados_df)

    # Calcular la efectividad del modelo
    accuracy = (y_test == y_pred).mean()

    # Crear DataFrame con valores únicos de y_test e y_pred
    unique_values_df = pd.DataFrame({
        'Categoría': pd.concat([y_test, pd.Series(y_pred)]).unique(),
        'Efectividad': f'{accuracy:.2%}'
        })

    # Depuración: Mostrar los valores únicos en y_test e y_pred en tres columnas con gráfica
    st.subheader('Valores de Efectividad Por Categorías')
    category_accuracies = {}
    for category in y_test.unique():
        category_mask = (y_test == category)
        category_accuracy = accuracy_score(y_test[category_mask], y_pred[category_mask])
        category_accuracies[category] = f'{category_accuracy:.2%}'

    # Crear DataFrame con efectividad para cada categoría
    category_accuracy_df = pd.DataFrame(list(category_accuracies.items()), columns=['Categoría', 'Efectividad'])

    # Mostrar el DataFrame utilizando st.table
    st.table(category_accuracy_df)

    # Crear la matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Crear un DataFrame de la matriz de confusión para mejorar la visualización
    conf_df = pd.DataFrame(conf_matrix, index=svm_model.classes_, columns=svm_model.classes_)

    # Graficar la matriz de confusión con seaborn
    st.subheader('Matriz de Confusión del Modelo SVM')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_df, annot=True, cmap='Blues', fmt='g', ax=ax)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicciones')
    plt.ylabel('Etiquetas Reales')
    st.pyplot(fig)

    # Calcular el reporte de clasificación
    classification_rep = classification_report(y_test, y_pred, target_names=svm_model.classes_)

    # Mostrar el reporte de clasificación en Streamlit
    st.subheader('Reporte de Clasificación del Modelo SVM')
    st.text(classification_rep)

    # Visualiza las distribuciones de las clases
    st.subheader('Distribución de Clases en Datos de Prueba Real')
    st.bar_chart(y_test.value_counts())

    # Añade Información de Depuración
    st.subheader('Información Adicional de Depuración')
    st.write('Aquí te muestro el 20% de todas las noticias que fueron usadas para probar el modelo')

    # Verifica la Predicción en un Ejemplo Aleatorio
    random_example_index = st.slider('Elije un ejemplo aleatorio', 0, len(X_test) - 1, 0)  # Slider para elegir el índice aleatorio
    st.subheader('Verificación de Predicción en un Ejemplo Aleatorio')
    st.write("Texto:", X_test.iloc[random_example_index])
    st.write("Etiqueta Real:", y_test.iloc[random_example_index])
    st.write("Predicción del Modelo:", y_pred[random_example_index])

    # Añadir caja de texto para ingresar contenido de la noticia
    st.subheader('Pon a prueba el modelo')
    st.write('Copia y pega el contenido de una noticia de: https://www.elsalvador.com/ de cualquier categoría y el modelo te dirá su predicción')
    contenido_noticia = st.text_area("Ingresa el contenido de la noticia:", "")

    # Botón para generar predicción
    if st.button("Generar Predicción"):
        # Realizar predicción en el contenido de la noticia ingresado
        contenido_noticia_tfidf = vectorizer.transform([contenido_noticia])
        contenido_noticia_selected = selector.transform(contenido_noticia_tfidf)
        prediccion_noticia = svm_model.predict(contenido_noticia_selected)[0]

        # Mostrar la predicción
        st.subheader('Predicción para la Noticia Ingresada')
        st.write(f'Predicción: {prediccion_noticia}')

# Página 3: Modelos NLP
elif pagina_seleccionada == "Modelos NLP":

    # Obtener las categorías disponibles
    categorias_disponibles = df_noticias['Categoría'].unique()

    # Obtener la entrada del usuario para seleccionar una categoría específica
    categoria_seleccionada = st.selectbox("Seleccione la categoría que desea analizar:", categorias_disponibles)

    # Filtrar las noticias por la categoría seleccionada
    noticias_categoria_seleccionada = df_noticias[df_noticias['Categoría'] == categoria_seleccionada]

    # Mostrar las noticias disponibles en la categoría seleccionada en forma de lista
    st.write(f"\nNoticias disponibles para la categoría '{categoria_seleccionada}':")
    if not noticias_categoria_seleccionada.empty:
        noticias_lista = noticias_categoria_seleccionada['Título'].tolist()
        seleccion_noticia = st.selectbox("Seleccione una noticia para analizar:", noticias_lista)
        indice_noticia = noticias_lista.index(seleccion_noticia)
    else:
        st.write("No hay noticias disponibles para esta categoría.")

    # Llamar a la función para generar la gráfica con la noticia específica
    titulo_noticia, frecuencia_entidades = procesar_noticia_seleccionada(noticias_categoria_seleccionada, indice_noticia, modelo_spacy)

    # Mostrar información en Streamlit
    st.write(f'\nNoticia: {titulo_noticia}')
    st.bar_chart(frecuencia_entidades)

    # Sección para ingresar texto personalizado
    st.header("Pon a prueba el modelo para Analizar Texto Personalizado")
    st.write('Puede buscar cualquier noticia de la página de El Salvador: https://www.elsalvador.com/')
    texto_personalizado = st.text_area("Ingrese el texto de la noticia:")
    if st.button("Analizar Texto"):
        if texto_personalizado:
        # Procesar el texto ingresado por el usuario
            doc_personalizado = modelo_spacy(texto_personalizado)
        
        # Obtener las entidades nombradas del texto
            entidades_personalizado = [ent.label_ for ent in doc_personalizado.ents]

        # Contar la frecuencia de cada tipo de entidad
            frecuencia_entidades_personalizado = {ent: entidades_personalizado.count(ent) for ent in set(entidades_personalizado)}

        # Mostrar la gráfica con la frecuencia de entidades del texto personalizado
            st.write("Entidades Nombradas en el Texto Personalizado:")
            st.bar_chart(frecuencia_entidades_personalizado)
        else:
            st.warning("Por favor, ingrese un texto antes de analizar.")

# Ejecutar la aplicación
if __name__ == '__main__':
    st.write('¡Gracias por visitar mi streamlit!, para más información de los códigos de los modelos visita mi repositorio de github: https://github.com/e2alvarado/Modelos-EDA-ML-NLP')
    st.write('En el enlace porporcionado podrás descargar los archivos e instalar toda la paquetería necesaria para ejecutarlo.')
    