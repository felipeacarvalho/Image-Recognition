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
'''
class EpochProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, canvas):
        super().__init__()
        self.canvas = canvas

    def on_epoch_begin(self, epoch, logs=None):
        self.canvas.current_epoch = epoch + 1
        self.canvas.num_epochs = self.params['epochs']

    def on_epoch_end(self, epoch, logs=None):
        if self.canvas.num_epochs > 0:
            progress_ratio = self.canvas.current_epoch / self.canvas.num_epochs
        else:
            progress_ratio = 0

        nSegmentos = int(progress_ratio * 20)

        print(f"Razão de progresso: {progress_ratio:.2f}, Segmentos: {nSegmentos}")

        self.canvas.drawProgressBar(self.canvas.progressbarSprite, progress_ratio)
'''
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

        self.cord1BP = (300, 150)
        self.cord2BP = (self.cord1BP[0]+368, self.cord1BP[1]+58)

        self.tempoTreinamento = None
        self.nomeOtimizador = None
        self.botao1Fill, self.botao2Fill = 1, 1
        self.mouseX, self.mouseY = -1, -1
        self.botaoIniciarTreinamento, self.botaoSalvarPesos = False, False
        self.treinamentoFinalizado = False
        self.trainingThread = None

        self.num_epochs = 0
        self.current_epoch = 0

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
    '''
    TODO
    def drawProgressBar(self, canvas, razaoProgresso):
        barX, barY = 300, 150  

        print(f"Canvas dimensions: {canvas.shape}")

        if self.progressbarSprite is None or self.progressbarSprite.shape[0] == 0 or self.progressbarSprite.shape[1] == 0:
            print("Error: Invalid progress bar sprite dimensions.")
            return

        bar_height, bar_width = self.progressbarSprite.shape[:2]


        print(f"Placing progress bar at ({barX}, {barY}) with size ({bar_width}, {bar_height})")

        if barX + bar_width > canvas.shape[1] or barY + bar_height > canvas.shape[0]:
            print(f"Error: progressbarSprite does not fit within the canvas at coordinates ({barX}, {barY}).")
            return

        try:
            canvas[barY:barY + bar_height, barX:barX + bar_width] = self.progressbarSprite
        except ValueError as e:
            print("Error during placement of progress bar:", e)
            return

        nSegmentos = int(razaoProgresso * 20)
        largSegmento = 15
        altSegmento = 29
        distSegmento = 2

        posXInicialSegmento = 316
        posYInicialSegmento = 165

        print(f"First segment position: ({posXInicialSegmento}, {posYInicialSegmento})")
        print(f"Number of segments to draw: {nSegmentos}")

        if self.progressSegmentSprite is None or \
        self.progressSegmentSprite.shape[0] < altSegmento or \
        self.progressSegmentSprite.shape[1] < largSegmento:
            print("Error: Invalid progress segment sprite dimensions.")
            return

        for i in range(nSegmentos):
            posX = posXInicialSegmento + i * (largSegmento + distSegmento)
            posY = posYInicialSegmento

            print(f"Segment {i+1} position: ({posX}, {posY})")

            if posX + largSegmento > canvas.shape[1] or posY + altSegmento > canvas.shape[0]:
                print(f"Skipping segment {i+1}: Exceeds canvas bounds at ({posX}, {posY}).")
                continue

            try:
                canvas[posY:posY + altSegmento, posX:posX + largSegmento] = \
                    self.progressSegmentSprite[0:altSegmento, 0:largSegmento]
            except ValueError as e:
                print(f"Error during placement of segment {i+1}:", e)
                break
    '''

    def treinModelo(self):
        try:
            tInicio = time.time()

            imgsTreinamento, nomesTreinamento = mc.carregarDadosTreinamento(self.pastaImg, self.txtNomes)
            self.modeloConstruido = mc.construirModelo()

            self.modeloConstruido.compile(optimizer='adam', 
                                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
                                 metrics=['accuracy'])

            self.num_epochs = 25
            '''
            epoch_callback = EpochProgressCallback(self)

            history = self.modeloConstruido.fit(
                imgsTreinamento, 
                nomesTreinamento, 
                epochs=self.num_epochs,
                callbacks=[epoch_callback]
            )
            '''
            history = self.modeloConstruido.fit(
                imgsTreinamento, 
                nomesTreinamento, 
                epochs=self.num_epochs
            )

            self.modeloConstruido.save_weights('weights/garrafas.weights.h5')

            self.history = history.history
            self.epocasTreinamento = range(1, len(history.history['accuracy']) + 1)
            self.tempoTreinamento = time.time() - tInicio
            self.treinamentoFinalizado = True


        except Exception as e:
            print("Erro durante execução do treinamento:", e)

    def exe(self):
        self.treinamentoFinalizado = False
        desBarraProgresso = False

        while True:    
            if self.botaoIniciarTreinamento:
                desBarraProgresso = True

            largCanvas, altCanvas = 900, 600
            canvas = np.ones((altCanvas, largCanvas, 3), dtype=np.uint8) * 24

            canvas[self.cord1B3[1]:self.cord2B3[1], self.cord1B3[0]:self.cord2B3[0]] = self.bImg1
            canvas[self.cord1B4[1]:self.cord2B4[1], self.cord1B4[0]:self.cord2B4[0]] = self.bImg2

            '''
            TODO
            if desBarraProgresso and not self.treinamentoFinalizado:
                if self.num_epochs > 0:
                    progress_ratio = self.current_epoch / self.num_epochs
                else:
                    progress_ratio = 0
                self.drawProgressBar(canvas, progress_ratio)

            if desBarraProgresso:
                canvas[self.cord1BP[1]:self.cord2BP[1], self.cord1BP[0]:self.cord2BP[0]] = self.progressbarSprite
            '''

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
                    self.modeloConstruido.save_weights('Image-Recognition/weights/garrafas.weights.h5')
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
