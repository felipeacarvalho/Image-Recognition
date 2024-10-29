import cv2
import numpy as np

largM, altM = 10, 300
larg1, alt1 = 400, 300
larg2, alt2 = 400, 300
larg, alt = 800, 300

Mar = np.zeros((altM, largM, 3), dtype = np.uint8)
canvas1 = np.ones((alt1, larg1, 3), dtype = np.uint8) * 255
canvas2 = np.ones((alt2, larg2, 3), dtype = np.uint8) * 127
canvas = np.zeros((alt, larg, 3), dtype = np.uint8)

canvas1_2 = np.hstack((Mar,canvas1, Mar, canvas2, Mar))

cv2.imshow('canvas12', canvas1_2)
#cv2.imshow('canvas2', canvas2)
cv2.imshow('canvas', canvas)

cv2.waitKey(0)
cv2.destroyAllWindows