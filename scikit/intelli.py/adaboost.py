import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


train = pd.read_csv("scikit/intelli.py/train.csv")
# print(train)
test = pd.read_csv("scikit/intelli.py/test.csv")
# print(test)

df = pd.concat([train, test], sort=False)
# print(df)
# print(df.info())
# print(df.isnull().sum())
df = df.drop('Cabin', axis=1)
# print(df)

# print(f"no. of null value befor impu:{df.Age.isnull().sum()}")
si = SimpleImputer(strategy='mean')
df['Age'] = si.fit_transform(df[['Age']])
# print(f"no. of null value after impu:{df.Age.isnull().sum()}")

df['Survived'] = si.fit_transform(df[['Survived']])
si2 =SimpleImputer(strategy='most_frequent')
df['Embarked'] = si2.fit_transform(df[['Embarked']]).ravel()
# print(df.isnull().sum())

# print(df.drop(['Name', 'Ticket'], axis=1))
threshold = 3
nf = ['Fare', 'Age']
z_scores = np.abs(zscore(df[nf]))
# print(z_score)
outliers = np.where(z_scores > threshold)
# print(outliers)

df['Age'] = winsorize(df['Age'], limits=[0.15,0.15])
# print(df['Age'])
df['Fare'] = winsorize(df['Fare'], limits=[0.15,0.15])
# print(df['Fare'])

oe = OrdinalEncoder()
df['Sex'] =oe.fit_transform(df[['Sex']])
# print(df['Sex'])

x = df.drop('Survived', axis=1)
y = df['Survived']
x1,x2,y1,y2 = train_test_split(x,y, test_size=0.30, random_state=0)
ab = AdaBoostClassifier(n_estimators=45,learning_rate=1, random_state=0)
ab.fit(x1,y1)
p = ab.predict(x2)
print(p)
ac = accuracy_score(p,y2)
print(ac)