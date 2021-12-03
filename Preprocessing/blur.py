import cv2
import random
import numpy as np

segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=1.5, k=300, min_size=5000)

src = cv2.imread('Input/dog.jpg')

segment = segmentator.processImage(src)

mask = segment.reshape(list(segment.shape) + [1]).repeat(3, axis=2)

masked = np.ma.masked_array(src, fill_value=1)

for i in range(np.max(segment)):
  masked.mask = mask != i
  y, x = np.where(segment == i)
  top, bottom, left, right = min(y), max(y), min(x), max(x)
  dst = masked.filled()[top : bottom + 1, left : right + 1]
  cv2.imwrite('Segment_Out/dog/{num}.jpg'.format(num=i), dst)