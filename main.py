from LinearRegression import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('Salary_Data.csv')
X=data.iloc[:,:-1].values
y=data.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=0)

model= LinearRegression()
J,theta=model.train(X_train,y_train)
plt.scatter(x=X,y=y)
x0 = np.linspace(0.75,10.5,100).reshape((100,1))
y0 = model.predict(x0)
plt.plot(x0, y0, '-r', label='y=X*theta')
plt.title('Graph of y=X*theta')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()
