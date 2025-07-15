import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer

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
df['embarked']=impf.fit_transform(df[['embarked']])

