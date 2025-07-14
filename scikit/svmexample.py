from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

x = iris.data
y = iris.target
classes = ['Iris setosa', 'Iris versicolour', 'Iris virginica']
print(x.shape, y.shape)


x1,x2,y1,y2 = train_test_split(x,y, test_size=0.2)
m = svm.SVC()
m.fit(x1,y1)
print(m)

print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape)


#prediction and accuracy

p = m.predict(x2)
acc = accuracy_score(y2,p)
print("prediction:", p)
print("accuravy:", acc)
