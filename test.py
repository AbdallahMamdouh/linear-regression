#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading data
data=pd.read_csv('Salary_Data.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values

#splitting data to train data and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)

#gradient descend
w=0.0
b=0.0
alpha=0.001
epochs=15000
data=pd.read_csv('Salary_Data.csv')
X=data['YearsExperience']
y=data['Salary']
N=len(X)
def update_w_b(X,y,w,b,alpha):
    dl_dw=0.0
    dl_db=0.0
    for i in range(N):
        dl_dw+=-2*X[i]*(y[i]-(w*X[i]+b))
        dl_db += -2 * (y[i] - (w * X[i] + b))
    w-=(1/float(N))*alpha*dl_dw
    b-=(1/float(N))*alpha*dl_db
    return w,b
def train(X,y,w,b,alpha,epochs):
    for e in range(epochs):
        w,b=update_w_b(X,y,w,b,alpha)
        if e%400==0:
            print("epoch:", e, "loss: ", avg_loss(X, y, w, b))
    return w,b
def avg_loss(spendings, sales, w, b):
    N = len(spendings)
    total_error = 0.0
    for i in range(N):
        total_error += (sales[i] - (w*spendings[i] + b))**2
    return total_error / float(N)
def predict(x,w,b):
    return w*x+b
w,b=train(X,y,w,b,alpha,epochs)
print(w,b)
x_new = 2.7
y_new=predict(x_new,w,b)
print(y_new)

#visualizing the result
plt.scatter(x=X,y=y)
x0 = np.linspace(0,10,100)
y0 = w*x0+b
plt.plot(x0, y0, '-r', label='y=wx+b')
plt.title('Graph of y=wx+b')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()
