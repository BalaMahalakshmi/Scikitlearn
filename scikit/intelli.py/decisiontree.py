import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


# df = pd.read_csv("scikit/intelli.py/bank.csv")
# # print(df)
# print(df.isna().sum())

# print(df[df.duplicated()])

# print(df.info())
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)
# print(df)

x = df['Age'].value_counts()
# print(x)
a = sns.countplot(data=df, x="Age")
# plt.show()

# print(df['diabetes.csv'].unique())

plt.figure(figsize=(8,8))
sns.displot(df.Age, color='green', label='Age', kde=True)
plt.legend()
# plt.show()

oe = OrdinalEncoder()
df['Age'] = oe.fit_transform(df[['Age']])

x = df.iloc[:,0:-1]
y = df.iloc[:,-1]
x1,x2,y1,y2 = train_test_split(x,y, test_size=0.2,random_state=42)

cg = DecisionTreeClassifier(criterion='gini', random_state=0)
cg.fit(x1,y1)
pred = cg.predict(x2)
# print(pred)

ac = accuracy_score(y2,pred)
# print(ac)

plt.figure(figsize=(8,8))
tree.plot_tree(cg.fit(x1,y1))
# plt.show()

centro = DecisionTreeClassifier(criterion='entropy',max_depth=3, random_state=0)
centro.fit(x1,y1)
p2 = centro.predict(x2)
# print(p2)
a2 = accuracy_score(y2, p2)
print("accuracy_score:" , a2)

plt.figure(figsize=(8,8))
tree.plot_tree(centro.fit(x1,y1))
plt.show()

