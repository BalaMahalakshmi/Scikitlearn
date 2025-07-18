import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("scikit/intelli.py/computer_purchase.csv")
# print(df)
# print(df.isna().sum())

b = df['Age'].value_counts()
# print(b)
a = sns.countplot(data=df, x="Age")
# plt.show()

b = df['Income'].value_counts()
# print(b)
a = sns.countplot(data=df, x="Income")
# plt.show()


plt.figure(figsize=(8,8))
sns.displot(df.Age, color='green', label='Age', kde=True)
plt.legend()
# plt.show()


cols = ['Age', 'Income', 'Student', 'CreditRating']
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])
df['BuysComputer'] = le.fit_transform(df['BuysComputer'])
x = df.drop('BuysComputer', axis=1)
y = df['BuysComputer']

x1,x2,y1,y2 = train_test_split(x,y, test_size=0.2,random_state=42)

cg = DecisionTreeClassifier(criterion='gini', random_state=0)
cg.fit(x1,y1)
pred = cg.predict(x2)
print("prediction:", pred)
ac = accuracy_score(y2,pred)
print("accuracy_score:" ,ac)

plt.figure(figsize=(8,8))
tree.plot_tree(cg.fit(x1,y1))
# plt.show()

centro = DecisionTreeClassifier(criterion='entropy',max_depth=3, random_state=0)
centro.fit(x1,y1)
p2 = centro.predict(x2)
# # print(p2)
a2 = accuracy_score(y2, p2)
# print("accuracy_score:" , a2)

# plt.figure(figsize=(8,8))
# tree.plot_tree(centro.fit(x1,y1))
# plt.show()
res = pd.DataFrame({'Actual': y2, 'Predicted': p2})
plt.figure(figsize=(8,6))
sns.countplot(data=res, x='Actual', hue='Predicted', palette='Set2')
plt.title('Actual vs Predicted (Countplot)')
plt.xlabel('Actual Labels')
plt.ylabel('Count')
plt.legend(title='Predicted')
plt.tight_layout()
# plt.show()
