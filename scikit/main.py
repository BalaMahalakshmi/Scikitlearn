import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
from sklearn.datasets import load_wine
from sklearn import datasets
import seaborn as sns
import math
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import LabelEncoder

# data = pd.read_csv("scikit\car.data")
# print(data)
# x = data[['A','B','C','D','E','F']].values
# y = data[['G']]
# print(x,y)

# Le = LabelEncoder()
# for i in range(len(x[0])):
#     x[:,i] = Le.fit_transform(x[:,i])
# #print(x)

# y = y['G'].replace({
#     'unacc':0,
#     "acc":1,
#     'good':2,
#     'vgood':3

# })

# x1,x2,y1,y2 = train_test_split(x,y, test_size=0.2, random_state=1)
# model = neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform')
# model.fit(x1,y1)
# p1 = model.predict(x2)
# a1 = metrics.accuracy_score(y2, p1)
# print("predict:",p1)
# print(a1)


data = datasets.load_wine(as_frame=True)
# print(data)

x = data.data
y = data.target
name = data.target_names
# print(name)

df = pd.DataFrame(x, columns=data.feature_names)
df['wc'] = data.target
df['wc'] = df['wc'].replace(to_replace=[0,1,2], value=['class0', 'class1', 'class2'])
sns.pairplot(data = df, hue='wc', palette='Set2')
x1,x2,y1,y2 = train_test_split(x,y, test_size=0.2, random_state=1)

print(math.sqrt(len(y2)))
model = neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform')
model.fit(x1,y1)
p1 = model.predict(x2)
a1 = metrics.accuracy_score(y2, p1)
print("predict:",p1)
print(a1)
k2 = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
k2.fit(x1,y1)
p2 = k2.predict(x2)
print(p2)


