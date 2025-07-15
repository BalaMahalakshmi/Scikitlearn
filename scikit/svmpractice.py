import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn import datasets
from sklearn import neighbors,metrics
from sklearn import svm

data = pd.read_csv("scikit\wine.data")
# print(data)

x = data.iloc[:,1:]
y = data.iloc[:,0]
x1,x2,y1,y2 = train_test_split(x, y, test_size=0.25, random_state=42)

m = svm.SVC(kernel='linear')
m.fit(x1,y1)
# print(m)

p = m.predict(x2)
acc = accuracy_score(y2,p)
 
# print("prediction:",p)
# print("accuracy:",acc)
# print("confusion matrix:",confusion_matrix(y2,p))

m2 = svm.SVC(kernel='rbf', random_state=1)
m2.fit(x1,y1)
print(m2)
p2 = m2.predict(x2)
acc2 = accuracy_score(y2,p)
print(p2)
print(acc2)
