import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import numpy
import os
import cv2
import gradio as gr
from GenerarDatos import IMAGE_SIZE, CATEGORIAS
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = 0.333)
#sess = tf.compat.v1.Session(config= tf.compat.v1.ConfigProto(gpu_options = gpu_options))
DATADIR3 = "..\\Insectos\\Dataset\\Test"
Trainx = pickle.load(open("TrainX.pickle","rb"))
Trainy = pickle.load(open("TrainY.pickle","rb"))
Valx = pickle.load(open("ValX.pickle","rb"))
Valy = pickle.load(open("ValY.pickle","rb"))
Testx = pickle.load(open("TestX.pickle","rb"))
Testy = pickle.load(open("TestY.pickle","rb"))
Trainx = Trainx / 255.0
Valx = Valx / 255.0
Testx = Testx / 255.0
Trainy = numpy.array(Trainy)
Valy = numpy.array(Valy)
Testy = numpy.array(Testy)

def prepare1(dir):
  
    img = cv2.cvtColor(dir, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

def prepare(dir):
    img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    
def predecir_sub(aux):     
        pred = tf.keras.models.load_model("models/Modelo3clases_2")
        info = pred.predict([prepare1(aux)])
        print(info)
        if(info[0][0] >= info[0][1]) and (info[0][0] >= info[0][2]):
            return CATEGORIAS[0]
        elif (info[0][1] >= info[0][0]) and (info[0][1] >= info [0][2]):
            return CATEGORIAS[1]
        else:
            return CATEGORIAS[2]
            
def predecir_mio(aux):
        print(aux)
        return "Esto era " + predecir_sub(aux)

def predecir():   
        pred = tf.keras.models.load_model("models/Modelo3clases_2")
        output = pred.evaluate(Testx, Testy)
        print("")
        print("=== Evaluation ===")
        print(pred.metrics_names)
        print(output)
        for categoria in CATEGORIAS:
            path = os.path.join(DATADIR3, categoria)
            valor = CATEGORIAS.index(categoria)
            fotos = os.listdir(path)
            for i in fotos: 
                info = pred.predict([prepare(os.path.join(path,i))])
                print("Esto era " + categoria)
                print("IMG: " + i)
                print("% " + CATEGORIAS[0] + " " + str(info[0][0]))
                print("% " + CATEGORIAS[1] + " " + str(info[0][1]))
                print("% " + CATEGORIAS[2] + " " + str(info[0][2]))
                #print("% " + CATEGORIAS[3] + " " + str(pred.predict([prepare(os.path.join(os.path.join(DATADIR3, categoria),i))])[0][3]))
                #print("% " + CATEGORIAS[4] + " " + str(pred.predict([prepare('esco.jpg')])[0][4]))
                print("------------------------------------")
                # print(CATEGORIAS[int(pred.predict([prepare('5406795-PPT.jpg')])[0][0])])
        return "Precision: " + str(output[1]) + "\n" + "para mas informacion en la consola"
               

def entrenar():
                    NAME = "Modelo3clases_2"
                    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
                    model = Sequential()
                    model.add(Conv2D(300, (3,3), input_shape = Trainx.shape[1:]))             
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))                                           
                    model.add(Flatten())
                    model.add(Dense(100))
                    model.add(Activation('relu')) 
                    model.add(Dropout(0.5))                     
                    model.add(Dense(100))
                    model.add(Activation('relu')) 
                    model.add(Dropout(0.5))                     
                    model.add(Dense(4))
                    model.add(Activation('softmax'))                  
                    model.compile(loss="sparse_categorical_crossentropy",metrics=['accuracy'])
                    early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=15)
                    history = model.fit(Trainx,Trainy, batch_size = 40, epochs = 60,  validation_data=(Valx,Valy),  callbacks=[early_stop],shuffle=True)
                    model.save("models/{}".format(NAME))
                    plt.plot(history.history['loss'], label='train loss')
                    plt.plot(history.history['val_loss'], label='val loss')
                    plt.legend()

                   #plt.savefig('vgg-loss-rps-1.png')
                    plt.show()
                
with gr.Blocks() as demo:
    name = gr.Image(shape=(200, 200))
    greet_btn = gr.Button("Start")
    greet_btn2 = gr.Button("Usar Dataset de prueba")
    output = gr.Textbox(label="Output Box")
    greet_btn.click(fn=predecir_mio, inputs=name, outputs=output)
    greet_btn2.click(fn=predecir,outputs=output)
demo.launch()
#entrenar()

