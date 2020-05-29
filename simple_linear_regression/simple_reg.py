#!/usr/bin/python3

# Simple Linear Regression using scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# import dataset and read it using pandas
fname = 'Salary_Data.csv'
df = pd.read_csv(fname)

# get the independent (X) and dependent (y) vars
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# split the dataset into training and testing tests using scikitlearn's 
# train_test_split() function
X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                test_size=0.20, random_state=0)

# create an object from the LinearRegression class of the scikit-learn library
regressor = LinearRegression()
reg_fit = regressor.fit(X_train,y_train)    # fit the training data
# because we want the number of the test set, we predict on the test set
y_pred = regressor.predict(X_test)  # y_pred contains the predicted salaries
print(f'\n\nPrediction:\n{y_pred}')

# visualizing the results, real salaries vs. predicted salaries (train set)
plt.scatter(X_train,y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visualizing the results, real salaries vs. predicted salaries (test set)
plt.scatter(X_test,y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# making a single prediction (e.g. the salary of an employee with 5 years of 
# experience)
# [[]] because the predict() expects a 2d arr
years_exp = 5
salary_emp = (regressor.predict([[years_exp]]))
print(f"\nThe salary of the employee with {years_exp} years of experience is: {salary_emp}")

# final linear regression eq. with the values of the coefficients
print(f"\nRegressor coefficient: {regressor.coef_}")
print(f"\nRegressor intercept/constant: {regressor.intercept_}")
