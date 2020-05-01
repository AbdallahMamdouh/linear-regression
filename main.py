from LinearRegression import LinearRegression, featureScale
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np, math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# loading data
data = pd.read_csv('Salary_Data.csv')
data = featureScale(data)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# splitting data to test and train sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# creating the model
model = LinearRegression()

# training the model
MSE, theta = model.train(X_train, y_train, Lambda=0.01)

# using the model to predict
y_pred = model.predict(X_test)

# calculating R2 score for my model
R2 = r2_score(y_test, y_pred)
print("R2 score = ", R2)

# plotting cost function
plt.grid()
plt.plot(MSE)
plt.show()

# calculating mean square values and root mean square
RMSE = math.sqrt(MSE[-1])
print("mean square error = ", MSE[-1])
print("root mean square error = ", RMSE)
