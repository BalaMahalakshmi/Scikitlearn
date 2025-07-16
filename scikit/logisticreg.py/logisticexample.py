import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

td = fetch_openml('titanic', version=1, as_frame=True)
df = td['data']
df['survived'] = td['target']
# print(df.head())

# print(sns.countplot(x='survived', data=df))
# print(sns.countplot(x='survived', hue= 'sex',data=df))
# plt.show()

df['age'].plot.hist()
# plt.show()

# df.info()
df.isnull().sum()
mv = pd.DataFrame(df.isnull().sum()/len(df)*100)
mv.plot(kind='bar', title='missing_value', ylabel='percentage')
# plt.show()

df ['family'] = df['sibsp']+df['parch']
df.loc[df['family']>0, 'travelled_alone']=0
df.loc[df['family']==0, 'travelled_alone']=1
# print(df['family'].head())

df.drop(['sibsp', 'parch'], axis=1, inplace=True)
sns.countplot(x='travelled_alone', data=df)
plt.title("no.of passengers travelling alone")
# plt.show()

df.drop(['name', 'ticket', 'home.dest'], axis=1, inplace=True)
# print(df.head())

df.drop(['cabin', 'body', 'boat'], axis=1, inplace=True)
# print(df.head())

s = pd.get_dummies(df['sex'], drop_first=True)
# print(s)

df['sex'] = s
# print(df.isnull().sum())

imp = SimpleImputer(strategy='mean')
df['age'] = imp.fit_transform(df[['age']])
df['fare'] = imp.fit_transform(df[['fare']])
# print(df.isnull().sum())

impf = SimpleImputer(strategy='most_frequent')
df['embarked']=impf.fit_transform(df[['embarked']]).ravel()
# print(df['embarked']) 

# print(df.isnull().sum())

# print(df.head())

e = pd.get_dummies(df['embarked'], drop_first=True)
# print(e)

df.drop(['embarked'], axis=1, inplace=True)
df=pd.concat([df,e],axis=1)
# print(df.head())

x = df.drop(['survived'], axis=1) 
y = df['survived']
# print(y.head())

x1,x2,y1,y2 = train_test_split(x,y, test_size=0.3,random_state=1)
# print(x1.shape, y1.shape)
# print(x2.shape, y2.shape)

m = LogisticRegression()
m.fit(x1,y1)

p = m.predict(x2)
print("predict:",p)
ac = accuracy_score(y2,p)
print(ac)
c = confusion_matrix(y2,p)
print(c)

