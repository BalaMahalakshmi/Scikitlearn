from sklearn.datasets import load_wine
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd

wine = load_wine()
x = pd.DataFrame(wine.data, columns=wine.feature_names)
y  = x['alcohol']      # target: alcohol content
X = x.drop(columns=['alcohol'])

lprac = linear_model.LinearRegression()
plt.scatter(x.iloc[:,0], y)
plt.xlabel(x.columns[0])
plt.ylabel('Alcohol content')
plt.show()

x1,x2,y1,y2 = train_test_split(x,y, test_size=0.2)
m = lprac.fit(x1,y1)
p = m.predict(x2)
print("prediction;",p)
print("R^2 value:", lprac.score(x,y))
print("coded:",lprac.coef_)
print("intercept:",lprac.intercept_)
