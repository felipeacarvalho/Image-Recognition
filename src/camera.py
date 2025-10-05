import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import Label
from selectTk import SelecDir as sd
import os
import time
import numpy as np

class Camera:

    def __init__(self):
        self.rX1, self.rY1 = 300, 610
        self.rX2, self.rY2 = 600, 640
        #self.cob= input('Cobaia: ')
        #global cob 
        self.bClick = False
        self.cobInputAt = False
        self.cobInput = ''

        self.cap = cv2.VideoCapture(0)

        cv2.namedWindow('PBCamera')

        cv2.createTrackbar('Margem', 'PBCamera', 100, 255, self.na)
        cv2.createTrackbar('Ilum.', 'PBCamera', 255, 255, self.na)
        #cv2.createTrackbar('Alpha', 'frame', 10, 100, self.na)
        cv2.createTrackbar('Contraste', 'PBCamera', 75, 100, self.na)

        cv2.setMouseCallback('PBCamera', self.mouseEv)

        self.bImg = cv2.resize(cv2.imread('assets/folder95.png'), (75, 75), interpolation = cv2.INTER_LINEAR)

    def na(self, x):
        pass

    def mouseEv(self, evento, x, y, flags, param):
        global dir
        bX1, bY1 = 100, 500
        bX2, bY2 = 175, 575

        if evento == cv2.EVENT_LBUTTONDOWN:
            if bX1 <= x <= bX2 and bY1 <= y <= bY2:
                dir = sd.expArq()

            if self.rX1 <= x <= self.rX2 and self.rY1 <= y <= self.rY2:
                self.cobInputAt = True
            else:
                self.cobInputAt = False

    def exe(self):

        contImg = 0

        i = 0
        cont = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            largMV, altMV = 10, 480
            MarV = np.ones((altMV, largMV, 3), dtype = np.uint8) * 24

            largMH, altMH = 1310, 10
            MarH = np.ones((altMH, largMH, 3), dtype = np.uint8) * 24

            largMH2, altMH2 = 1310, 200
            MarH2 = np.ones((altMH2, largMH2, 3), dtype = np.uint8) * 24
       
            ###

            margemV = cv2.getTrackbarPos('Margem', 'PBCamera')
            maxV = cv2.getTrackbarPos('Ilum.', 'PBCamera')
            #alphaV = cv2.getTrackbarPos('Alpha', 'PBCamera')
            conV = ((cv2.getTrackbarPos('Contraste', 'PBCamera')+10)/50)

            ###

            pbCap = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            conCap = cv2.convertScaleAbs(pbCap, 1, conV)
            (_, PBCamera) = cv2.threshold(conCap, margemV, maxV, cv2.THRESH_TOZERO)
            PBCamera_cor = cv2.cvtColor(PBCamera, cv2.COLOR_GRAY2BGR)

            ###

            canvasH = np.hstack((MarV, PBCamera_cor, MarV, frame, MarV))
            canvasV = np.vstack((MarH, canvasH, MarH2))
            
            bX1, bY1 = 100, 500
            bX2, bY2 = 175, 575
            canvasV[bY1:bY2, bX1:bX2] = self.bImg

            cv2.putText(canvasV, ('Nome da cobaia:'), (300, 600), cv2.FONT_HERSHEY_PLAIN, 1.4, (255, 255, 255), 1)
            cv2.rectangle(canvasV, (self.rX1, self.rY1), (self.rX2, self.rY2), (255, 255, 255), 1)
            cv2.putText(canvasV, self.cobInput, (310, 632), cv2.FONT_HERSHEY_PLAIN, 1.4, (255, 255, 255), 1)
            
            cv2.putText(canvasV, (f'Caminho Selecionado: {dir}'), (300, 550), cv2.FONT_HERSHEY_PLAIN, 1.4, (255, 255, 255), 1)

            
            cv2.putText(canvasV, (f'Contador: {contImg}'), (650, 632), cv2.FONT_HERSHEY_PLAIN, 1.4, (255, 255, 255), 1)
            
            cv2.imshow('PBCamera', canvasV)

            #if (contImg % 100 == 0 and contImg != 0):
                #print("Próxima cobaia:")
                #cob = input("")

            key = cv2.waitKey(1) & 0xFF

            if self.cobInputAt:
                if key == 8:
                    self.cobInput = self.cobInput[:-1]
                elif key == 13:
                    print(f'Input: {self.cobInput}')
                    self.cobInputAt = False
                    #self.cobInput = ''
                elif 32<= key <= 126:
                    self.cobInput += chr(key)
            if key == ord('f'):
                i = 1
                cont = 1


            elif key == ord('e'):
                i = 0
                cont = 0

            if i == 1 or cont == 1:

                try:
                    nomeImg = '{}'.format(self.cobInput) + '{}.png'.format(contImg+1)
                    #nomeImg = 'img{}.png'.format(contImg)
                    cv2.imwrite(os.path.join(dir, nomeImg), PBCamera)

                    print('{} escrito.'.format(nomeImg))

                    contImg += 1
                    i = 0
                    time.sleep(0.7)
                        
                except:
                    nomeImg = 'default' + '{}.png'.format(contImg+1)
                    #nomeImg = 'img{}.png'.format(contImg)
                    cv2.imwrite(os.path.join(dir, nomeImg), PBCamera)
                    print('{} escrito.'.format(nomeImg))
                    contImg += 1

            elif key == 27:
                break


        self.cap.release()
        cv2.destroyAllWindows()

camera = Camera()
camera.exe()
       