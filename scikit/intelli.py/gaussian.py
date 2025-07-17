import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)
# print(df)

# print(df.head())

print(df.info())

print(df[df.duplicated()])
print(df.drop_duplicates(keep='first',inplace=True))

x = (df.Pregnancies.value_counts())
print(f"no. of pregnancies {x[1]} and no. of non pregnancies {x[0]}")
a = sns.countplot(data=df, x="Pregnancies")
plt.show()


x = (df.BloodPressure.value_counts())
# print(x)
a = sns.countplot(data=df, x="BloodPressure")
# plt.show()

plt.figure(figsize=(8,8))
sns.displot(df.Age, color='green', label='age', kde=True)
plt.legend()
plt.show()


plt.figure(figsize=(8,8))
sns.displot(df[df['Outcome']==0]['Age'], color='green', label='increasing bp', kde=True)
sns.displot(df[df['Outcome']==1]['Age'], color='red', label='taking insulin', kde=True)
plt.title("BMI vs Insulin")
plt.show()

x = df.iloc[:,-1].values
y = df.iloc[:,-1].values
x = x.reshape(-1,1)

x1,x2,y1,y2 = train_test_split(x,y, test_size=0.25,random_state=42)
sc = StandardScaler()
x1 =sc.fit_transform(x1)
x2 = sc.transform(x2)

gb = GaussianNB()
gb.fit(x1,y1)
pred = gb.predict(x2)
# print(pred)
pred_proba = gb.predict_proba(x2)
# print(pred_proba)
ac = accuracy_score(pred,y2)
print(ac)

