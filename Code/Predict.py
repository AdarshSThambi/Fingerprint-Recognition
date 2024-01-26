

import Minutiae
import tensorflow as tf
# import numpy as np

import numpy
model = tf.keras.models.load_model('CNN_fingerprint.h5')

image_path = '../Data/Test/5/5_4.tif'
img_dim = (388, 388)

im_matrix = numpy.zeros(img_dim)
minutiae_list = Minutiae.compute_minutiae(image_path)
for point in minutiae_list:
    im_matrix[point[0]][point[1]] = 1

im_matrix = numpy.reshape(im_matrix, (1, 388, 388, 1))

prediction = model.predict(im_matrix)

class_index = numpy.argmax(prediction)

print("Predicted class is:", class_index)