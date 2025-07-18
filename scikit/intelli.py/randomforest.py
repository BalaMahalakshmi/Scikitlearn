import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from yellowbrick.model_selection import validation_curve

df = pd.read_csv("scikit/wine.data")
# print(df.head())
# print(df.columns)

cn = ['Alcohol','Malic acid', 'Ash', 'Alcalinity of ash ', 'Magnesium', 'Total Phenols', 'Flavanoids', 
      'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline ']
df.columns =cn
# print(df.info())
# print(df.describe(include='all').T)

# for col in cn:
    # print(df[col].value_counts())
# print(df[df.duplicated()])

# oe = OrdinalEncoder()
# df['Alcohol'] = oe.fit_transform(df[['Alcohol']])
# df['Ash'] = oe.fit_transform(df[['Ash']])
# df['Malic acid'] = oe.fit_transform(df[['Malic acid']])
# df['Alcalinity of ash'] = oe.fit_transform(df[['Alcalinity of ash']])
# df['Magnesium'] = oe.fit_transform(df[['Magnesium']])
# df['Proline'] = oe.fit_transform(df[['Proline']])

# print(df['Alcohol'])

x = df.iloc[:, 0:-1]
y = df.iloc[:,-1]
x1,x2,y1,y2 = train_test_split(x,y, test_size=0.3,random_state=0)
rfc = RandomForestClassifier()
rfc.fit(x1,y1)
p1 = rfc.predict(x2)
# print(p1)
ac1 = accuracy_score(p1,y2)
# print(ac1)

num =[100,200,450,500,750,1000]
# print(validation_curve(RandomForestClassifier(), X=x1, y=y1, param_name="n_estimators", param_range=num, scoring="accuracy", cv=3))

dep_val =[10,7,15,12,20,22]
# print(validation_curve(RandomForestClassifier(), X=x1, y=y1, param_name="max_depth", param_range=dep_val, scoring="accuracy", cv=3))

min_val =[8,5,3,6,12,4]
# print(validation_curve(RandomForestClassifier(), X=x1, y=y1, param_name="min_samples_split", param_range=min_val, scoring="accuracy", cv=3))

rfc2 = RandomForestClassifier(n_estimators=1000, min_samples_split=3, max_depth=15, random_state=0)
rfc2.fit(x1,y1)
p2 = rfc2.predict(x2)
print(p2)
ac = accuracy_score(p2, y2)
# print("accuracy_score:"ac)

feature_score = pd.Series(rfc2.feature_importances_, index = x1.columns).sort_values(ascending=False)
# print(feature_score)

sns.barplot(x = feature_score, y = feature_score.index)
plt.xlabel("feature importance score")
# plt.show()

rfc3 = RandomForestClassifier()
xn = df.drop(['Ash','Magnesium'], axis=1)
yn = df['Proline ']
x1,x2,y1,y2 = train_test_split(x,y, test_size=0.3)

rfc3.fit(x1,y1)
p3 = rfc3.predict(x2)
print(p3)
ac3 = accuracy_score(p3,y2)
print("accuracyscore:",ac3)