#pip install pillow mnist numpy sklearn

from PIL import Image
import mnist
import numpy as np
from sklearn.neural_network import  MLPClassifier
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import mnist
(x1,y1), (x2,y2) = mnist.load_data()

x1 = mnist.train_images()
y1 = mnist.train_labels()
x2 = mnist.test_images
y2 = mnist.test_labels()

print("x_train:",x1)
print("x_test:",x2)
print("y_train:",y1)
print("y_test:",y2)
