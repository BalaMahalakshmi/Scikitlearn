import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("scikit\car.data")
print(data)
x = data[['A','B','C','D','E','F']].values
y = data[['G']]
print(x,y)

Le = LabelEncoder()
for i in range(len(x[0])):
    x[:,i] = Le.fit_transform(x[:,i])
#print(x)

y = y['G'].replace({
    'unacc':0,
    "acc":1,
    'good':2,
    'vgood':3

})

x1,x2,y1,y2 = train_test_split(x,y, test_size=0.2)
model = neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform')
model.fit(x1,y1)
p1 = model.predict(x2)
a1 = metrics.accuracy_score(y2, p1)
print("predict:",p1)
print(a1)
