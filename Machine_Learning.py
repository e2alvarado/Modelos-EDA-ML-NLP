import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Descargar recursos de NLTK (si no se han descargado previamente)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Obtener la fecha actual
fecha_actual = datetime.now().strftime('%d-%m-%Y')

# Ruta del archivo CSV generado por scrape-diario.py con la fecha actual
ruta_csv_scrape_diario = f'Noticias_{fecha_actual}.csv'

# Leer el DataFrame desde el archivo CSV
df_noticias = pd.read_csv(ruta_csv_scrape_diario)

# Definir funciones de limpieza y preprocesamiento
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text, categories_to_remove):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if word not in categories_to_remove]
    return " ".join(tokens)

# Obtener las categorías y crear una lista de etiquetas a eliminar
categorias = df_noticias['Categoría'].unique()
etiquetas_a_eliminar = [cat.lower() for cat in categorias]

# Aplicar la limpieza y preprocesamiento al contenido de las noticias
df_noticias['Contenido_limpio'] = df_noticias.apply(lambda row: clean_text(row['Contenido'], etiquetas_a_eliminar), axis=1)

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df_noticias['Contenido_limpio'], df_noticias['Categoría'], test_size=0.2, random_state=42)

# Vectorización del texto usando TF-IDF con un número limitado de características
vectorizer = TfidfVectorizer(max_features=1500)  # Ajusta el número de características según sea necesario
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Selección de características utilizando chi-cuadrado
selector = SelectKBest(chi2, k=1000)  # Ajusta el número de características según sea necesario
X_train_selected = selector.fit_transform(X_train_tfidf, y_train)
X_test_selected = selector.transform(X_test_tfidf)

# Ajuste de hiperparámetros y entrenamiento del modelo SVM
svm_model = SVC(kernel='linear', C=0.2)  # Ajusta el valor de C según sea necesario
svm_model.fit(X_train_selected, y_train)

# Predicciones y evaluación del modelo
y_pred = svm_model.predict(X_test_selected)
print(classification_report(y_test, y_pred))





import pickle

# Guardar las variables necesarias
with open('variables_modelo.pkl', 'wb') as f:
    pickle.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'vectorizer': vectorizer,
        'selector': selector,
        'svm_model': svm_model
    }, f)





# Crear la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Crear un DataFrame de la matriz de confusión para mejorar la visualización
conf_df = pd.DataFrame(conf_matrix, index=categorias, columns=categorias)

# Graficar la matriz de confusión con seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(conf_df, annot=True, cmap='Blues', fmt='g')
plt.title('Matriz de Confusión')
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas Reales')
plt.show()




# Crear un DataFrame con las métricas por categoría
metrics_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T

# Graficar las métricas utilizando seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x=metrics_df.index, y=metrics_df['f1-score'])
plt.title('Puntuación F1 por Categoría')
plt.xlabel('Categoría')
plt.ylabel('Puntuación F1')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



#MODELO 2 ML

import pandas as pd
from gensim import corpora, models
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import string

# Obtener el contenido de las noticias del DataFrame
news_corpus = df_noticias['Contenido'].tolist()

# Tokenización y preprocesamiento
stop_words = set(stopwords.words('spanish'))  # Palabras vacías en español
punctuation = set(string.punctuation)  # Puntuaciones

tokenized_news = []
for news in news_corpus:
    # Tokenizar el texto y convertir a minúsculas
    tokens = word_tokenize(news.lower())

    # Remover palabras vacías y puntuaciones
    filtered_tokens = [word for word in tokens if word not in stop_words and word not in punctuation]

    tokenized_news.append(filtered_tokens)

# Crear un diccionario a partir del corpus
dictionary = corpora.Dictionary(tokenized_news)

# Crear un corpus Gensim
corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_news]

# Entrenar el modelo LDA
lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# Obtener la representación de tópicos para cada documento
topic_representation = [lda_model[doc] for doc in corpus]

# Convertir la representación de tópicos a un formato adecuado
topics_list = []
for topics in topic_representation:
    topic_dict = dict(topics)
    topics_list.append([topic_dict.get(i, 0) for i in range(lda_model.num_topics)])

# Crear un DataFrame con las representaciones de tópicos
topics_df = pd.DataFrame(topics_list, columns=[f'Topic_{i}' for i in range(lda_model.num_topics)])

# Utilizar técnicas de clustering para segmentar la audiencia
num_clusters = 3  # Ajusta según sea necesario
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(topics_df)

# Agregar la columna de clusters al DataFrame original
df_noticias['Cluster'] = clusters

# Ver los resultados de la segmentación
print(df_noticias[['Contenido', 'Cluster']])





import pickle

# Guardar el modelo LDA
lda_model.save('lda_model')

# Guardar el corpus
with open('corpus.pkl', 'wb') as f:
    pickle.dump(corpus, f)

# Guardar el diccionario
dictionary.save('dictionary')





from gensim import corpora, models

# Cargar el modelo LDA
lda_model = models.LdaModel.load('lda_model')

# Cargar el corpus
with open('corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)

# Cargar el diccionario
dictionary = corpora.Dictionary.load('dictionary')



#distribucion de cada tópico
import matplotlib.pyplot as plt

topic_distribution = [max(doc, key=lambda x: x[1])[0] for doc in lda_model[corpus]]
plt.hist(topic_distribution, bins=range(lda_model.num_topics + 1), align='left', rwidth=0.8)
plt.xlabel('Tópico')
plt.ylabel('Número de Documentos')
plt.title('Distribución de Tópicos en Documentos')
plt.xticks(range(lda_model.num_topics), [f'Tópico {i}' for i in range(lda_model.num_topics)], rotation=45)
plt.show()

#visualizar las palabras mas usadas en cada tópico
from wordcloud import WordCloud

for idx, topic_words in lda_model.print_topics():
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(topic_words)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Topic {idx} - Word Cloud')
    plt.show()



#Representa el documento mas representativo de cada topico
import numpy as np

topic_document_matrix = np.zeros((len(corpus), lda_model.num_topics))
for doc_id, doc in enumerate(corpus):
    for topic_id, topic_prob in lda_model[doc]:
        topic_document_matrix[doc_id, topic_id] = topic_prob

most_representative_docs = np.argmax(topic_document_matrix, axis=0)
for topic_id, doc_id in enumerate(most_representative_docs):
    print(f"Most representative document for Topic {topic_id}: Document {doc_id}")


#MODELO 3 ML
    
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

# Supongamos que tenemos una lista de palabras únicas de las noticias
lista_palabras = " ".join(df_noticias['Contenido']).split()

modelo_word2vec = Word2Vec(sentences=[lista_palabras], vector_size=100, window=5, min_count=1, workers=4)
modelo_word2vec.train([lista_palabras], total_examples=len(lista_palabras), epochs=10)

# Visualizar un ejemplo de relación entre palabras usando embeddings
palabras_interesantes = ['noticia', 'información', 'evento', 'categoría', 'autor']
vectores_interesantes = [modelo_word2vec.wv[palabra] for palabra in palabras_interesantes]

plt.figure(figsize=(8, 6))

# Graficar los vectores de palabras
for i, palabra in enumerate(palabras_interesantes):
    plt.scatter(vectores_interesantes[i][0], vectores_interesantes[i][1], label=palabra)
    plt.text(vectores_interesantes[i][0], vectores_interesantes[i][1], palabra, fontsize=9)

plt.xlabel('Dimensión 1')
plt.ylabel('Dimensión 2')
plt.title('Relaciones Semánticas entre Palabras con Word Embeddings')
plt.grid()
plt.legend()
plt.show()
