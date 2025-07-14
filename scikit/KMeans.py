from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd
from sklearn import metrics

bc = load_breast_cancer()
# print(bc)

x = scale(bc.data)
# print(x)
y = bc.target

#creating a model

x1,x2,y1,y2 = train_test_split(x,y, test_size=6)
m = KMeans(n_clusters=2, random_state=0)
m.fit(x1)

p = m.predict(x2)
labels = m.labels_
# print("labels:",labels)
# print("predictions:",p)
# print("accuracy:",accuracy_score(y2,p))
# print("actual:",y2)
# print(len(y2), len(labels))
# print(pd.crosstab(y1,labels))

def bench_k_means(estimator, name, data):
    to_time()
    estimator.fit(data)

bench_k_means(model,'1', x)
