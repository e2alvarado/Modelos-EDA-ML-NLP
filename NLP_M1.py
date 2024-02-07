import spacy
import pandas as pd

def cargar_modelo():
    return spacy.load('en_core_web_sm')

def obtener_entidades_noticia(texto_noticia, modelo):
    doc = modelo(texto_noticia)
    entidades = [ent.label_ for ent in doc.ents]
    frecuencia_entidades = {ent: entidades.count(ent) for ent in set(entidades)}
    return frecuencia_entidades

def procesar_noticia_seleccionada(df_noticias, indice_noticia, modelo_spacy):
    texto_noticia = df_noticias['Contenido'].iloc[indice_noticia]
    titulo_noticia = df_noticias['Título'].iloc[indice_noticia]
    frecuencia_entidades = obtener_entidades_noticia(texto_noticia, modelo_spacy)
    return titulo_noticia, frecuencia_entidades
