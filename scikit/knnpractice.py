import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn import neighbors,metrics
from matplotlib import pyplot as plt


data = pd.read_csv("scikit\wine.data")
print(data)
print(data.columns)
data.columns = list('ABCDEFGHIJKLMN')



x = data[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']].values

y = data[['N']]
# print(x,y)

Le = LabelEncoder()
for i in range(len(x[0])):
    x[:,i] = Le.fit_transform(x[:,i])
# print(x)

x1,x2,y1,y2 = train_test_split(x,y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(x1)
X_test = scaler.transform(x2)


model = neighbors.KNeighborsClassifier(n_neighbors=3)
model.fit(x1,y1)
p1 = model.predict(x2)
a1 = metrics.accuracy_score(y2, p1)
print("predict:",p1)
print("confusion matrix:" ,confusion_matrix(y2,p1))
print("accuracy:",a1)

#plot the graph
neighbors_list = [1,3,5,7,9,11]
accuracies = []

plt.figure(figsize=(8, 5))
plt.bar(neighbors_list, accuracies, color='skyblue')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN: Accuracy vs Number of Neighbors')
plt.show()

 
