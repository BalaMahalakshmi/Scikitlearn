from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


digits = load_digits()
x = digits.data
y = digits.target
# print(df_digits.head())

df = pd.DataFrame(x, columns=digits.feature_names)
df['digits class'] = y
# print(df)

# print(df.isnull().sum())

# print(df.describe())

sc = StandardScaler()
x = sc.fit_transform(x)
# print(x)
wss=[]
for i in range(1,10):
    KM = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    KM.fit(x)
    wss.append(KM.inertia_)

f, ax = plt.subplots(figsize=(8,6))
plt.plot(range(1,10),wss)
plt.title("technique")
plt.xlabel("no.of clusters")
plt.ylabel("wss-kmeans")
# plt.show()

N = 4
km = KMeans(init= 'k-means++', n_clusters=N)
km.fit(x)
labels = km.labels_
# print(labels)

ac = accuracy_score(labels,y)
print(ac)