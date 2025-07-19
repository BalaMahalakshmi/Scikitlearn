import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


df = pd.read_csv("scikit/car.data")
# print(df.head())
# print(df.shape)
data = df.iloc[:200]
# print(data.shape)
# print(df.info())
le = LabelEncoder()
data['B'] = le.fit_transform(data['B'])
# print(data['B'])
data['C'] = le.fit_transform(data['C'])
# print(data['C'])
data['D'] = le.fit_transform(data['D'])
# print(data['D'])
# print(data.info())

x = data.drop(['E','G'], axis=1)
y = data['G']
for col in x.columns:
    if x[col].dtype=='object':
        x[col]=le.fit_transform(x[col])

x1,x2,y1,y2=train_test_split(x,y,test_size=0.2, random_state=42)
rf = RandomForestClassifier()
rf.fit(x1,y1)
p=rf.predict(x2)
# print(p)
ac = accuracy_score(p,y2)
# print(ac)

forest_params ={'max_depth': list(range(10,15)), 'max_features': list(range(0,14))}
gs = GridSearchCV(rf, forest_params, cv=10, scoring='accuracy')
gs.fit(x1,y1)
# print(gs.best_params_)
# print(gs.best_score_) 
p=rf.predict(x2)
print(p)
ac = accuracy_score(p,y2)
print(ac)
