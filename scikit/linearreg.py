from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing



# boston = datasets.load_boston()
# x = boston.data
# y = boston.target
# print(x.shape, y.shape)


# Use California housing dataset instead
california_housing = fetch_california_housing()
# print(california_housing)
x = california_housing.data
y = california_housing.target
# print(x.shape, y.shape)


#model creation

lreg = linear_model.LinearRegression()
plt.scatter(x.T[3], y)
plt.show()

x1,x2,y1,y2 = train_test_split(x,y, test_size=0.2)
m = lreg.fit(x1,y1)
p = m.predict(x2)
print("prediction;",p)
print("R^2 value:", lreg.score(x,y))
print("coded:",lreg.coef_)
print("intercept:",lreg.intercept_)

