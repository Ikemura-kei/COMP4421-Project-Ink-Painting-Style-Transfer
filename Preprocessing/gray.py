import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.2989, 0.5870, 0.1140])

def layering(img):
    output = img
    for i in range(len(img)):
        for j in range(len(img[0])):
            output[i][j] = img[i][j] * 255 // 32 * 32 / 255
    return output

img = mpimg.imread('Input/scene.png')     
gray = rgb2gray(img)
layer = layering(gray)

im = Image.fromarray(layer*255)
if im.mode != 'L':
    im = im.convert('L')
im.save('Output/scene_layer.jpg')

plt.imshow(layer, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.show()