#pip install pillow
#pip install opencv-python
#pip install numpy

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import Label
from selectTk import SelecArq as sa
import tensorflow as tf
import os
import time
import numpy as np
import modeloClass as mc

class Camera:

    def __init__(self):
        self.r1X1, self.r1Y1 = 700, 50
        self.r1X2, self.r1Y2 = 920, 100

        self.r2X1, self.r2Y1 = 700, 150
        self.r2X2, self.r2Y2 = 920, 200

        self.r3X1, self.r3Y1 = 50, 510
        self.r3X2, self.r3Y2 = 250, 560

        self.carregado = False

        self.iniRecon = False

        self.pesos = 'weights/rede.weights.h5'

        self.fill1 = 1
        self.fill2 = 1

        self.mouseX, self.mouseY = -1, -1

        self.cap = cv2.VideoCapture(0)

        cv2.namedWindow('PBCamera')

        cv2.createTrackbar('Margem', 'PBCamera', 100, 255, self.na)
        cv2.createTrackbar('Ilum.', 'PBCamera', 255, 255, self.na)
        #cv2.createTrackbar('Alpha', 'frame', 10, 100, self.na)
        cv2.createTrackbar('Contraste', 'PBCamera', 75, 100, self.na)

        cv2.setMouseCallback('PBCamera', self.mouseEv)

    def na(self, x):
        pass

    def mouseEv(self, evento, x, y, flags, param):

        if evento == cv2.EVENT_MOUSEMOVE:
            self.mouseX, self.mouseY = x, y

        if evento == cv2.EVENT_LBUTTONDOWN:
            if self.r1X1 <= x <= self.r1X2 and self.r1Y1 <= y <= self.r1Y2:
                self.iniRecon = True
                self.carregado = False

            elif self.r2X1 <= x <= self.r2X2 and self.r2Y1 <= y <= self.r2Y2:
                self.iniRecon = False

            else:
                pass

    def exe(self):

        erroArqPesos = False
        dirTmp = 'tmp'
        recon = False

        while True:

            if (self.r1X1 <= self.mouseX <= self.r1X2 and self.r1Y1 <= self.mouseY <= self.r1Y2):
                self.fill1 = -1
            else:
                self.fill1 = 1

            if (self.r2X1 <= self.mouseX <= self.r2X2 and self.r2Y1 <= self.mouseY <= self.r2Y2):
                self.fill2 = -1
            else:
                self.fill2 = 1

            ret, frame = self.cap.read()
            if not ret:
                break

            largMV, altMV = 10, 480
            MarV = np.ones((altMV, largMV, 3), dtype=np.uint8) * 24

            largMV, altMV = 310, 480
            MarV2 = np.ones((altMV, largMV, 3), dtype=np.uint8) * 24

            largMH, altMH = 960, 10
            MarH = np.ones((altMH, largMH, 3), dtype=np.uint8) * 24

            largMH2, altMH2 = 960, 100
            MarH2 = np.ones((altMH2, largMH2, 3), dtype=np.uint8) * 24

            ###

            margemV = cv2.getTrackbarPos('Margem', 'PBCamera')
            maxV = cv2.getTrackbarPos('Ilum.', 'PBCamera')
            conV = ((cv2.getTrackbarPos('Contraste', 'PBCamera') + 10) / 50)

            ###

            pbCap = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            conCap = cv2.convertScaleAbs(pbCap, 1, conV)
            (_, PBCamera) = cv2.threshold(conCap, margemV, maxV, cv2.THRESH_TOZERO)
            PBCamera_cor = cv2.cvtColor(PBCamera, cv2.COLOR_GRAY2BGR)

            ###

            canvasH = np.hstack((MarV, PBCamera_cor, MarV2))
            canvasV = np.vstack((MarH, canvasH, MarH2))

            cv2.rectangle(canvasV, (self.r1X1, self.r1Y1), (self.r1X2, self.r1Y2), (185, 56, 31), self.fill1)
            cv2.putText(canvasV, "Iniciar reconhecimento", (715, 80), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            cv2.rectangle(canvasV, (self.r2X1, self.r2Y1), (self.r2X2, self.r2Y2), (15, 68, 252), self.fill2)
            cv2.putText(canvasV, "Pausar reconhecimento", (715, 180), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            cv2.rectangle(canvasV, (self.r3X1, self.r3Y1), (self.r3X2, self.r3Y2), (255, 255, 255), 1)
            try:
                cv2.putText(canvasV, nome_prev, (60, 540), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            except:
                pass

            if erroArqPesos:
                cv2.putText(canvasV, "Formato de arquivo de pesos invalido.", (50, 580), cv2.FONT_HERSHEY_PLAIN, 1, (26, 21, 179), 1)
            else:
                pass

            cv2.imshow('PBCamera', canvasV)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == 32:
                try:
                    self.pesos = sa.expArq()
                    erroArqPesos = False
                except:
                    self.pesos = 'weights/rede.weights.h5'

            
            if (self.iniRecon == True and self.carregado == False):
                try:
                    modelo = mc.construirModelo()
                    modelo = mc.carregarPesos(modelo, self.pesos)

                    img = cv2.resize(PBCamera, (640, 480))
                    imgProc = np.expand_dims(img, axis=0)
                    imgProc = imgProc.astype('float32') / 255.0 

                    previsao = mc.predict(modelo, imgProc)
                    classe_prev = np.argmax(previsao)
                    #nomes_classes = ['Felipe (oculos)', 'Felipe (s/ oculos)', 'Fernando (oculos)', 'Fernando (s/ oculos)', 'Gabriel S.', 'Gabriel D. (oculos)', 'Gabriel D. (s/ oculos)']
                    nomes_classes = ['Garrafa de Coca-Cola', 'Garrafa de Guaraná']
                    nome_prev = nomes_classes[classe_prev]

                    print(f"Reconhecimento: {nome_prev}")
                    #self.carregado = True

                except:
                    pass

            else:
                pass

        self.cap.release()
        cv2.destroyAllWindows()

camera = Camera()
camera.exe()
       