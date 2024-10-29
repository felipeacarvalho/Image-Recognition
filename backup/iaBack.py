#pip install matplotlib numpy keras tensorflow pillow

import cv2
import numpy as np
from selectTk import SelecArq as sa, SelecDir as sd
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import time
from carr import carregarImgNomes as cin
import modeloClass as mc
#from teste import Captura2

class CanvasIA:

    def __init__(self):
        cv2.namedWindow('JFORdCP')

        ####

        self.r1X1, self.r1Y1 = 150, 390
        self.r1X2, self.r1Y2 = 250, 430

        self.r2X1, self.r2Y1 = 150, 490
        self.r2X2, self.r2Y2 = 250, 530

        self.b1X1, self.b1Y1 = 50, 375
        self.b1X2, self.b1Y2 = 125, 450

        self.b2X1, self.b2Y1 = 50, 475
        self.b2X2, self.b2Y2 = 125, 550

        self.b3X1, self.b3Y1 = 50, 50
        self.b3X2, self.b3Y2 = 270, 100

        self.b4X1, self.b4Y1 = 50, 150
        self.b4X2, self.b4Y2 = 270, 200

        self.fill2, self.fill3 = 1, 1

        self.mouseX, self.mouseY = -1, -1

        ####

        self.bCar, self.bIni, self.bSal = False, False, False
        
        self.bImg1 = cv2.resize(cv2.imread('assets/folder95.png'), (75, 75), interpolation = cv2.INTER_LINEAR)
        self.bImg2 = cv2.resize(cv2.imread('assets/file95.png'), (75, 75), interpolation = cv2.INTER_LINEAR)

        cv2.setMouseCallback('JFORdCP', self.mouseEv)

        ####

    def mouseEv(self, evento, x, y, flags, param):
        global pastaImg
        global txtNomes

        if evento == cv2.EVENT_MOUSEMOVE:
            self.mouseX, self.mouseY = x, y

        if evento == cv2.EVENT_LBUTTONDOWN:
            if self.b1X1 <= x <= self.b1X2 and self.b1Y1 <= y <= self.b1Y2:
                pastaImg = sd.expArq()

            elif self.b2X1 <= x <= self.b2X2 and self.b2Y1 <= y <= self.b2Y2:
                txtNomes = sa.expArq()

            elif self.b3X1 <= x <= self.b3X2 and self.b3Y1 <= y <= self.b3Y2:
                self.bIni = True
                self.bSal = False
                #self.prov = True

            elif self.b4X1 <= x <= self.b4X2 and self.b4Y1 <= y <= self.b4Y2:
                self.bSal = True
                self.bIni = False


    ####

    def exe(self):

        erro1 = False
        self.prov = False
        self.treinamentoFinalizado = False
        erroDir1, erroDir2 = False, False
        desRetDir = False
        treinado, pesosSalvos = False, False

        modelo = None

        while True:    

            largCanvas, altCanvas = 800, 600
            canvas = np.ones((altCanvas, largCanvas, 3), dtype = np.uint8) * 24

            canvas[self.b1Y1:self.b1Y2, self.b1X1:self.b1X2] = self.bImg1
            canvas[self.b2Y1:self.b2Y2, self.b2X1:self.b2X2] = self.bImg2

            if (self.b3X1 <= self.mouseX <= self.b3X2 and self.b3Y1 <= self.mouseY <= self.b3Y2):
                self.fill2 = -1

            else:
                self.fill2 = 1

            if (self.b4X1 <= self.mouseX <= self.b4X2 and self.b4Y1 <= self.mouseY <= self.b4Y2):
                self.fill3 = -1

            else:
                self.fill3 = 1
            
            cv2.rectangle(canvas, (self.b3X1, self.b3Y1), (self.b3X2, self.b3Y2), (15, 68, 252), int(self.fill2))
            cv2.putText(canvas, ('Treinar rede'), (100, 80), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            if((erroDir1 == True or erroDir2 == True) and desRetDir == True):
                cv2.rectangle(canvas, (35, 365), (140, 560), (26, 21, 179), 2)
                cv2.putText(canvas, ('Selecione os diretorios de treinamento'), (50, 590), cv2.FONT_HERSHEY_PLAIN, 1, (26, 21, 179), 1)
            elif(erroDir1 == False and erroDir2 == False):
                desRetDir = False

            if(treinado == True):
                cv2.putText(canvas, ('Treinamento concluido.'), (50, 130), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            else:
                pass

            if(pesosSalvos == True):
                cv2.putText(canvas, ('Pesos da rede salvos.'), (50, 230), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            else:
                pass

            if (self.treinamentoFinalizado == True):

                cv2.rectangle(canvas, (self.b4X1, self.b4Y1), (self.b4X2, self.b4Y2), (15, 68, 252), int(self.fill3))
                cv2.putText(canvas, ('Salvar pesos da rede'), (60, 180), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            try:
                cv2.putText(canvas, (f'Diretorio selecionado: {pastaImg}'), (155, 420), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            except:
                pass

            try:
                cv2.putText(canvas, (f'Arquivo selecionado: {txtNomes}'), (155, 520), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            except:
                pass

            cv2.imshow('JFORdCP', canvas)

            try:
                pImg = pastaImg
            except:
                pass

            try:
                tNomes = txtNomes
            except:
                pass


            if (self.bIni == True and erro1 == False and self.prov == False):
                self.bIni = False

                start_time = time.time()

                try:
                    pImg = pastaImg
                    erroDir1 = False
                except:
                    pass

                try:
                    tNomes = txtNomes
                    erroDir2 = False
                except:
                    pass

                try:
                    imgs_treinamento, nomes_treinamento = cin(pImg, tNomes)

                    modeloC = mc.construirModelo()
                    modeloC.compile(optimizer='ftrl', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

                    history = modeloC.fit(imgs_treinamento, nomes_treinamento, epochs=60)

                    elapsed_time = time.time() - start_time

                    best_accuracy = max(history.history['accuracy'])
                    lowest_loss = min(history.history['loss'])
                    epochs_range = range(1, len(history.history['accuracy']) + 1)

                    self.treinamentoFinalizado = True

                    treinado = True

                except:
                    erroDir1, erroDir2 = True, True
                    desRetDir = True

            elif (self.bSal == True and self.treinamentoFinalizado == True):

                self.bSal = False

                modeloC.save_weights('weights/rede2.weights.h5')

                print('Pesos salvos')

                pesosSalvos = True

            try:
                if(any(pImg) and pImg != ''):
                    erroDir1 = False
            except:
                pass

            try:
                if(any(tNomes) and tNomes != ''):
                    erroDir2 = False
            except:
                pass

            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                try:

                    fig, ax1 = plt.subplots(figsize=(10, 6))

                    ax1.set_xlabel('Épocas')
                    ax1.set_ylabel('Erro', color='tab:red')
                    ax1.plot(epochs_range, history.history['loss'], label='Erro', color='tab:red')
                    ax1.tick_params(axis='y', labelcolor='tab:red')
                    ax1.set_title(f'Erro e Precisão no treinamento\nMenor erro: {lowest_loss:.4f}, Maior precisão: {best_accuracy:.4f}')

                    ax1.annotate(f'Menor erro: {lowest_loss:.4f}', 
                                xy=(history.history['loss'].index(lowest_loss) + 1, lowest_loss),
                                xytext=(history.history['loss'].index(lowest_loss) + 1.5, lowest_loss + 0.02),
                                arrowprops=dict(facecolor='black', shrink=0.05))
                    
                    ax2 = ax1.twinx()  
                    ax2.set_ylabel('Precisão', color='tab:blue')
                    ax2.plot(epochs_range, history.history['accuracy'], label='Precisão', color='tab:blue')
                    ax2.tick_params(axis='y', labelcolor='tab:blue')

                    ax2.annotate(f'Maior precisão: {best_accuracy:.4f}', 
                                xy=(history.history['accuracy'].index(best_accuracy) + 1, best_accuracy),
                                xytext=(history.history['accuracy'].index(best_accuracy) + 1.5, best_accuracy - 0.02),
                                arrowprops=dict(facecolor='black', shrink=0.05))

                    ax2.set_ylim(0, 1)
                    
                    timef = elapsed_time / 60

                    fig.suptitle(f"Tempo de treinamento: {timef:.2f} minutos", fontsize=16)

                    plt.tight_layout()
                    plt.show()

                except:
                    pass

            elif key == 27:
                break

        cv2.destroyAllWindows()

canvasIA = CanvasIA()
canvasIA.exe()
