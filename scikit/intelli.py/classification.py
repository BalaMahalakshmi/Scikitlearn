import numpy as np
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([1,0,0,0])
# print(x)
# print(y)
w = np.array([1,1])
print(w)
b = -0.6
# print(b)

def activation(z):
    if z>=0:
        return 1
    else:
        return 0


p = []
for a in x:
    y_hat = np.dot(a,w)+b
    p.append(activation(y_hat))
    # print(p)


import math 
epochs = 100
alpha = 0.2
w1 = np.random.random()
w2 = np.random.random()
w3 = np.random.random()
# print('w1:',w1, 'w2:',w2, 'w3:',w3)

del_w1 = 1
del_w2 = 1
del_w3 = 1

t = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
op = np.array([0,1,1,1,1,1,1,1])
print(t)
print(op)

bias = 0
for i in range (epochs):
    j =0
    for x in t:
        y_hat = w1*x[0]+w2*x[1]+w3*x[2]+bias
        if (y_hat>=0):
            act = 1
        else:
            act =0
        e = op[j] - act

del_w1 = alpha*x[0]*e
del_w2 = alpha*x[1]*e
del_w3 = alpha*x[2]*e

w1 = w1+del_w1
w2 = w2+del_w2
w3 = w3+del_w3

j =j=1
print("epochs", i+1, 'error:',e)
print(del_w1,del_w2,del_w3)
print(w1,w2,w3) 




