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

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, canvasIA, numEpocas):
        super().__init__()
        self.canvasIA = canvasIA
        self.numEpocas = numEpocas

    def on_epoch_end(self, epoch, logs=None):
        razaoProgresso = (epoch + 1) / self.numEpocas
        self.canvasIA.updateProgress(razaoProgresso)

class CanvasIA:

    def __init__(self):
        cv2.namedWindow('EoN')

        self.cord1B1 = (50, 50)
        self.cord2B1 = (270, 100)
        self.cord1B2 = (50, 150)
        self.cord2B2 = (270, 200)
        self.cord1B3 = (50, 375)
        self.cord2B3 = (125, 450)
        self.cord1B4 = (50, 475)
        self.cord2B4 = (270, 200)
        self.cord1B3 = (50, 375)
        self.cord2B3 = (125, 450)
        self.cord1B4 = (50, 475)
        self.cord2B4 = (125, 550)

        self.tempoTreinamento = None
        self.nomeOtimizador = None
        self.botao1Fill, self.botao2Fill = 1, 1
        self.mouseX, self.mouseY = -1, -1
        self.botaoIniciarTreinamento, self.botaoSalvarPesos = False, False
        self.treinamentoFinalizado = False
        self.trainingThread = None

        diretorioImagem = os.path.dirname(os.path.abspath(__file__))

        trainImgFolderPath = os.path.join(diretorioImagem, '../img')
        trainTxtFilePath = os.path.join(diretorioImagem, '../labels.txt')
        folderImgPath = os.path.join(diretorioImagem, '../assets/folder95.png')
        fileImgPath = os.path.join(diretorioImagem, '../assets/file95.png')
        progressbarImgPath = os.path.join(diretorioImagem, '../assets/progressbar95.png')
        progressSegmentImgPath = os.path.join(diretorioImagem, '../assets/progress95.png')

        self.pastaImg, self.txtNomes = trainImgFolderPath, trainTxtFilePath
        
        self.bImg1 = cv2.resize(cv2.imread(folderImgPath), (75, 75), interpolation=cv2.INTER_LINEAR)
        self.bImg2 = cv2.resize(cv2.imread(fileImgPath), (75, 75), interpolation=cv2.INTER_LINEAR)
        self.progressbarSprite = cv2.imread(progressbarImgPath)
        self.progressSegmentSprite = cv2.imread(progressSegmentImgPath)

        cv2.setMouseCallback('EoN', self.mouseEv)

    def createButton(self, canvas, cord1, cord2, texto: str, textX, textY, fillN) -> None:

        cv2.rectangle(canvas, cord1, cord2, (15, 68, 252), int(fillN))
        cv2.putText(canvas, texto, (textX, textY), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    def mouseEv(self, eventoMouse, x, y, flags, param):
        if eventoMouse == cv2.EVENT_MOUSEMOVE:
            self.mouseX, self.mouseY = x, y

        if eventoMouse == cv2.EVENT_LBUTTONDOWN:
            if self.cord1B1[0] <= x <= self.cord2B1[0] and self.cord1B1[1] <= y <= self.cord2B1[1]:
                self.botaoIniciarTreinamento = True
                self.botaoSalvarPesos = False
            elif self.cord1B2[0] <= x <= self.cord2B2[0] and self.cord1B2[1] <= y <= self.cord2B2[1]:
                self.botaoSalvarPesos = True
                self.botaoIniciarTreinamento = False
            elif self.cord1B3[0] <= x <= self.cord2B3[0] and self.cord1B3[1] <= y <= self.cord2B3[1]:
                self.pastaImg = sd.expArq() or ""
            elif self.cord1B4[0] <= x <= self.cord2B4[0] and self.cord1B4[1] <= y <= self.cord2B4[1]:
                self.txtNomes = sa.expArq() or ""

    def drawProgressBar(self, canvas, razaoProgresso):
        barX, barY = 126, 271

        canvas[barY:barY+self.progressbarSprite.shape[0], barX:barX+self.progressbarSprite.shape[1]] = self.progressbarSprite

        nSegmentos = int(razaoProgresso * 20)
        largSegmento = self.progressSegmentSprite.shape[1]

        for i in range(nSegmentos):
            posX = barX + i * largSegmento
            canvas[barY:barY + self.progressSegmentSprite.shape[0], posX:posX + largSegmento] = self.progressSegmentSprite

    def updateProgress(self, razaoProgresso):
        largCanvas, altCanvas = 800, 600
        canvas = np.ones((altCanvas, largCanvas, 3), dtype=np.uint8) * 24
        self.drawProgressBar(canvas, razaoProgresso)
        cv2.imshow('EoN', canvas)
        cv2.waitKey(1)

    def treinModelo(self):
        try:
            tInicio = time.time()

            imgsTreinamento, nomesTreinamento = mc.carregarDadosTreinamento(self.pastaImg, self.txtNomes)
            self.modeloConstruido = mc.construirModelo()

            self.modeloConstruido.compile(optimizer='adam', 
                                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
                                 metrics=['accuracy'])

            numEpocas = 3
            progress_callback = TrainingProgressCallback(self, numEpocas)

            history = self.modeloConstruido.fit(
                imgsTreinamento, 
                nomesTreinamento, 
                epochs=numEpocas, 
                callbacks=[progress_callback]
            )

            self.history = history.history
            self.epocasTreinamento = range(1, len(history.history['accuracy']) + 1)
            self.tempoTreinamento = time.time() - tInicio
            self.treinamentoFinalizado = True

        except Exception as e:
            print("Erro durante execução do treinamento:", e)

    def exe(self):
        self.treinamentoFinalizado = False

        while True:    
            largCanvas, altCanvas = 900, 600
            canvas = np.ones((altCanvas, largCanvas, 3), dtype=np.uint8) * 24

            canvas[self.cord1B3[1]:self.cord2B3[1], self.cord1B3[0]:self.cord2B3[0]] = self.bImg1
            canvas[self.cord1B4[1]:self.cord2B4[1], self.cord1B4[0]:self.cord2B4[0]] = self.bImg2

            self.botao1Fill = -1 if (self.cord1B1[0] <= self.mouseX <= self.cord2B1[0]  and self.cord1B1[1] <= self.mouseY <= self.cord2B1[1]) else 1
            self.botao2Fill = -1 if (self.cord1B2[0] <= self.mouseX <= self.cord2B2[0] and self.cord1B2[1] <= self.mouseY <= self.cord2B2[1]) else 1


            self.createButton(canvas, self.cord1B1, self.cord2B1, 'Treinar rede', 100, 80, self.botao1Fill)

            if self.treinamentoFinalizado:
                self.createButton(canvas, self.cord1B2, self.cord2B2, 'Salvar pesos da rede', 60, 180, self.botao2Fill)

            cv2.putText(canvas, f'Diretorio selecionado: {self.pastaImg}', (155, 420), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            cv2.putText(canvas, f'Arquivo selecionado: {self.txtNomes}', (155, 520), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            cv2.imshow('EoN', canvas)

            if self.botaoIniciarTreinamento and not self.treinamentoFinalizado:
                self.botaoIniciarTreinamento = False
                if not self.trainingThread or not self.trainingThread.is_alive():
                    self.trainingThread = threading.Thread(target=self.treinModelo)
                    self.trainingThread.start()

            if self.botaoSalvarPesos and self.treinamentoFinalizado:
                self.botaoSalvarPesos = False
                if hasattr(self, 'modeloConstruido'):
                    self.modeloConstruido.save_weights('Image-Recognition/weights/rede2.weights.h5')
                    print('Pesos salvos')
                    pesosSalvos = True
                else:
                    print("Modelo não treinado. Não é possível salvar pesos.")

            key = cv2.waitKey(1) & 0xFF
            if key == 32 and self.treinamentoFinalizado:
                maiorPrecisao = max(self.history['accuracy'])
                precisaoFinal = self.history['accuracy'][-1]
                menorErro = min(self.history['loss'])
                erroFinal = self.history['loss'][-1]

                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax1.set_xlabel('Épocas')
                ax1.set_ylabel('Erro', color='tab:red')
                ax1.plot(self.epocasTreinamento, self.history['loss'], label='Erro', color='tab:red')
                ax1.tick_params(axis='y', labelcolor='tab:red')

                ax2 = ax1.twinx()
                ax2.set_ylabel('Precisão', color='tab:blue')
                ax2.plot(self.epocasTreinamento, self.history['accuracy'], label='Precisão', color='tab:blue')
                ax2.tick_params(axis='y', labelcolor='tab:blue')
                ax2.set_ylim(0, 1)

                plt.title("Resumo do Treinamento da Rede Neural")
                plt.text(1, 0.6, 
                         f"Tempo de treinamento: {self.tempoTreinamento:.2f} segundos\n"
                         f"Otim. utilizado: {self.nomeOtimizador}\n"
                         f"Maior precisão: {maiorPrecisao:.2%}\n"
                         f"Menor erro: {menorErro:.4f}\n"
                         f"Precisão final: {precisaoFinal:.2%}\n"
                         f"Erro final: {erroFinal:.4f}",
                         fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

                plt.tight_layout()
                plt.show()

            elif key == 27:  
                break

        cv2.destroyAllWindows()

canvasIA = CanvasIA()
canvasIA.exe()
