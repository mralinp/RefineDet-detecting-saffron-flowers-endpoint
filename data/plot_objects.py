from matplotlib import pyplot as plt
import numpy as np
import cv2
from utils import *

def draw_line_segment(image, center, angle, color, length=20, thickness=2):
    x1 = center[0] - cosd(angle) * length / 2
    x2 = center[0] + cosd(angle) * length / 2
    y1 = center[1] - sind(angle) * length / 2
    y2 = center[1] + sind(angle) * length / 2

    cv2.line(image, (int(x1 + .5), int(y1 + .5)), (int(x2 + .5), int(y2 + .5)), color, thickness)

image = cv2.imread('Labeled/001.jpg')
flowers = np.loadtxt('Labeled/001.csv', delimiter=',')

draw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
for i in range(flowers.shape[0]):
    draw_line_segment(draw, flowers[i, :2], flowers[i, 2], (0, 255, 0))
    
print(flowers)

plt.imshow(draw)
plt.show()