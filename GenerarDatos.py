import os
import cv2
from tqdm import tqdm
import random as rn
import numpy as np
import pickle

def Generar_datos(aux,aux2,aux3):
    data = []
    filename = "..\\Insectos\\VisualizarDataset"
    for categoria in CATEGORIAS:
        path = os.path.join(aux, categoria)
        valor = CATEGORIAS.index(categoria)
        listdir = os.listdir(path)
        for i in tqdm(range(len(listdir)), desc = categoria):
            imagen_nombre = listdir[i]
            try:
                imagen_ruta = os.path.join(path, imagen_nombre)
                imagen = cv2.imread(imagen_ruta, cv2.IMREAD_GRAYSCALE) 
                imagen = cv2.resize(imagen,(IMAGE_SIZE, IMAGE_SIZE))
               #cv2.imwrite(os.path.join(filename, imagen_nombre + categoria + ".jpg"),imagen)
                data.append([imagen, valor])
            except Exception as e:
                pass
    rn.shuffle(data)
    x = []
    y = []

    for i in tqdm(range(len(data)),desc="Procesamiento"):
        par = data[i]
        x.append(par[0])
        y.append(par[1])

    x = np.array(x).reshape(-1,IMAGE_SIZE, IMAGE_SIZE,1)

    pickle_out = open(aux2,"wb")
    pickle.dump(x, pickle_out)
    pickle_out.close()
    print("Archivo " + aux2 + "creado!")

    pickle_out = open(aux3,"wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()
    print("Archivo " + aux3 + "creado!")


CATEGORIAS = ["Latrodectus","Mantis","Tijereta"] #,"Lithobius sp","Lycosidae","Mantis","Pandinus imperator","Trichonephila"
IMAGE_SIZE = 100

if __name__ == "__main__":
    DATADIR = "..\\PracticaFinalSI\\Insectos\\Dataset\\Train"
    DATADIR2 = "..\\Insectos\\Dataset\\Valid"
    DATADIR3 = "..\\Insectos\\Dataset\\Test"
    Generar_datos(DATADIR,"TrainX.pickle","TrainY.pickle")
    Generar_datos(DATADIR2,"ValX.pickle","ValY.pickle")
    Generar_datos(DATADIR3,"TestX.pickle","TestY.pickle")