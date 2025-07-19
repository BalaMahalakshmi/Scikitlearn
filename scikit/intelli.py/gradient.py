import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = datasets.load_diabetes()
# print(data)
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = pd.Series(data.target)
# print(df)

# print(df.info())
# print(df[df.duplicated()])
y = df['target']
x = df.drop('target', axis=1)
x1,x2,y1,y2 = train_test_split(x,y, test_size=0.2, random_state=0)
gbr= GradientBoostingRegressor()
gbr.fit(x1,y1)
p = gbr.predict(x2)
# print(p)

mae = mean_absolute_error(p,y2)
# print(mae)

feature_score = pd.Series(gbr.feature_importances_, index =x2.columns).sort_values(ascending=False)
sns.barplot(x=feature_score, y=feature_score.index)
plt.xlabel("feature importance score")
plt.ylabel("feature")
# plt.show()

test_score = np.zeros(500, dtype=np.float64)
for i,p in enumerate(gbr.staged_predict(x2)):
    test_score[i] = mean_squared_error(y2,p)
n_estimators = gbr.train_score_.shape[0]
fig = plt.figure(figsize=(10,10))
plt.subplot(1,1,1)
plt.title("Deviance")
plt.plot(np.arange(n_estimators)+1, gbr.train_score_, 'b-', label="Training set deviance")
plt.plot(np.arange(n_estimators)+1,test_score[:n_estimators], 'r-', label="Test deviance")
plt.legend(loc='upper right')
plt.xlabel("Boosting Iteration")
plt.ylabel("Deviance")
plt.tight_layout
plt.show()


