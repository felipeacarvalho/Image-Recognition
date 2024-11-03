# pip install matplotlib numpy keras tensorflow pillow opencv-python-headless

import cv2
import numpy as np
from selectTk import SelecArq as sa, SelecDir as sd
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import time
import modeloClass as mc
import threading
import os

class CanvasIA:

    def __init__(self):
        cv2.namedWindow('EoN')

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
        self.bCar, self.bIni, self.bSal = False, False, False
        self.treinamentoFinalizado = False
        self.trainingThread = None

        script_dir = os.path.dirname(os.path.abspath(__file__))

        trainImgFolderPath = os.path.join(script_dir, '../img')
        trainTxtFilePath = os.path.join(script_dir, '../labels.txt')

        folderImgPath = os.path.join(script_dir, '../assets/folder95.png')
        fileImgPath = os.path.join(script_dir, '../assets/file95.png')

        self.pastaImg, self.txtNomes = trainImgFolderPath, trainTxtFilePath
        
        self.bImg1 = cv2.resize(cv2.imread(folderImgPath), (75, 75), interpolation=cv2.INTER_LINEAR)
        self.bImg2 = cv2.resize(cv2.imread(fileImgPath), (75, 75), interpolation=cv2.INTER_LINEAR)

        cv2.setMouseCallback('EoN', self.mouseEv)

    def mouseEv(self, evento, x, y, flags, param):
        if evento == cv2.EVENT_MOUSEMOVE:
            self.mouseX, self.mouseY = x, y

        if evento == cv2.EVENT_LBUTTONDOWN:
            if self.b1X1 <= x <= self.b1X2 and self.b1Y1 <= y <= self.b1Y2:
                self.pastaImg = sd.expArq() or ""
            elif self.b2X1 <= x <= self.b2X2 and self.b2Y1 <= y <= self.b2Y2:
                self.txtNomes = sa.expArq() or ""
            elif self.b3X1 <= x <= self.b3X2 and self.b3Y1 <= y <= self.b3Y2:
                self.bIni = True
                self.bSal = False
            elif self.b4X1 <= x <= self.b4X2 and self.b4Y1 <= y <= self.b4Y2:
                self.bSal = True
                self.bIni = False

    def treinModelo(self):
        try:
            imgs_treinamento, nomes_treinamento = mc.carregarDadosTreinamento(self.pastaImg, self.txtNomes)
            self.modeloC = mc.construirModelo()
            self.modeloC.compile(optimizer='adam', 
                                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                                 metrics=['accuracy'])
            history = self.modeloC.fit(imgs_treinamento, nomes_treinamento, epochs=20)
            self.history = history.history
            self.epochs_range = range(1, len(history.history['accuracy']) + 1)
            self.treinamentoFinalizado = True
        except Exception as e:
            print("Erro durante execução do treinamento:", e)

    def exe(self):
        erro1 = False
        self.treinamentoFinalizado = False
        erroDir1, erroDir2 = False, False
        desRetDir = False
        treinado, pesosSalvos = False, False

        while True:    
            largCanvas, altCanvas = 800, 600
            canvas = np.ones((altCanvas, largCanvas, 3), dtype=np.uint8) * 24

            canvas[self.b1Y1:self.b1Y2, self.b1X1:self.b1X2] = self.bImg1
            canvas[self.b2Y1:self.b2Y2, self.b2X1:self.b2X2] = self.bImg2

            self.fill2 = -1 if (self.b3X1 <= self.mouseX <= self.b3X2 and self.b3Y1 <= self.mouseY <= self.b3Y2) else 1
            self.fill3 = -1 if (self.b4X1 <= self.mouseX <= self.b4X2 and self.b4Y1 <= self.mouseY <= self.b4Y2) else 1

            cv2.rectangle(canvas, (self.b3X1, self.b3Y1), (self.b3X2, self.b3Y2), (15, 68, 252), int(self.fill2))
            cv2.putText(canvas, 'Treinar rede', (100, 80), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            if self.treinamentoFinalizado:
                cv2.rectangle(canvas, (self.b4X1, self.b4Y1), (self.b4X2, self.b4Y2), (15, 68, 252), int(self.fill3))
                cv2.putText(canvas, 'Salvar pesos da rede', (60, 180), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            cv2.putText(canvas, f'Diretorio selecionado: {self.pastaImg}', (155, 420), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            cv2.putText(canvas, f'Arquivo selecionado: {self.txtNomes}', (155, 520), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            cv2.imshow('EoN', canvas)

            if self.bIni and not self.treinamentoFinalizado:
                self.bIni = False
                if not self.trainingThread or not self.trainingThread.is_alive():
                    self.trainingThread = threading.Thread(target=self.treinModelo)
                    self.trainingThread.start()

            if self.bSal and self.treinamentoFinalizado:
                self.bSal = False
                if hasattr(self, 'modeloC'):
                    self.modeloC.save_weights('Image-Recognition/weights/rede2.weights.h5')
                    print('Pesos salvos')
                    pesosSalvos = True
                else:
                    print("Modelo não treinado. Não é possível salvar pesos.")

            key = cv2.waitKey(1) & 0xFF
            if key == 32 and self.treinamentoFinalizado:
                best_accuracy = max(self.history['accuracy'])
                lowest_loss = min(self.history['loss'])
                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax1.set_xlabel('Épocas')
                ax1.set_ylabel('Erro', color='tab:red')
                ax1.plot(self.epochs_range, self.history['loss'], label='Erro', color='tab:red')
                ax1.tick_params(axis='y', labelcolor='tab:red')
                ax2 = ax1.twinx()
                ax2.set_ylabel('Precisão', color='tab:blue')
                ax2.plot(self.epochs_range, self.history['accuracy'], label='Precisão', color='tab:blue')
                ax2.tick_params(axis='y', labelcolor='tab:blue')
                ax2.set_ylim(0, 1)
                plt.tight_layout()
                plt.show()

            elif key == 27:  
                break

        cv2.destroyAllWindows()

canvasIA = CanvasIA()
canvasIA.exe()
