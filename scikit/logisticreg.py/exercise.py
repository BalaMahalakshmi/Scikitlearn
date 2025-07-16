import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)
# print(df)

# print(df.head())

# print(df.columns)

sns.countplot(x='Outcome', data=df)
# plt.show()
df['Age'].plot.hist()
# plt.show()

# print(df.info())
# print(df.isnull().sum())

mv = pd.DataFrame(df.isnull().sum()/len(df)*100)
mv.plot(kind='bar', title='missing_value', ylabel='affected percentage')
# plt.show()

df.drop(['Pregnancies', 'Insulin', 'BMI'], axis=1, inplace=True)
# print(df.head())

# s = pd.get_dummies(df['BMI'], drop_first=True)
# print(s)

# df['BMI'] = s
# print(df.isnull().sum())

x = df.drop('Outcome', axis=1)
y = df['Outcome']
x1,x2,y1,y2 = train_test_split(x,y, test_size=0.3,random_state=1)
print(x1.shape, y1.shape)
print(x2.shape, y2.shape)

m = LogisticRegression()
m.fit(x1,y1)

p = m.predict(x2)
print("predict:",p)
ac = accuracy_score(y2,p)
print(ac)
c = confusion_matrix(y2,p)
print(c)