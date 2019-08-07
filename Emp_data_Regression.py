#import all the modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#obtain the same split everytime you run your script
from sklearn.model_selection import train_test_split
#to split the data for training and testing.
from sklearn.linear_model import LinearRegression
#for builting the model...

data = pd.read_csv("company.csv", squeeze = True)

real_x = data.iloc[:,0].values
real_y = data.iloc[:,1].values
#convert into 2d array
#by using reshape function converts in 2D Array (-1 is used to for whole row,1 for first col )
real_x = real_x.reshape(-1,1)
real_y = real_y.reshape(-1,1)

# print(real_x)

training_x,testing_x,training_y,testing_y = train_test_split(real_x, real_y, test_size = 0.5, random_state = 0)
#test_size = 0.3 is data splited in 30% for testing.
#train_test_split =  split te data for testing and traning.
#random_state = match the data .

#for using linear Regression.
lin = LinearRegression()
#for fitting all the values.
lin.fit(training_x,training_y)

#for finding the predicted value in testing set .
pred_y = lin.predict(testing_x)

#for finding the predicted value in testing set .
pred_y = lin.predict(testing_x)

# main formula for prediction.
#  y = b1x + b0
# where b1 is COEFFICENT and b0 is INTERCEPT
# b1
print("coefficient of the data " ,lin.coef_)
# b0
print(" Intercept of the data " ,lin.intercept_)

# testing of data manually implementing values in formula
# 9360.26128619*10.5 + 26777.3913412

#give out the testing Value as per index.
# print(testing_y[8.2])

#gives out predicted value for index value in test data.
# print(pred_y[8])


#Plot the scatter graph
plt.scatter(training_x,training_y, color = "green")
#plot LinearRegression Line
plt.plot(training_x,lin.predict(training_x), color = "blue")
#Title,Labels resp:
plt.title("TRANING salary and expirence table")
plt.xlabel("Expirence")
plt.ylabel("Salary")
plt.show()


#Plot the scatter graph
plt.scatter(testing_x,testing_y, color = "green")
#plot LinearRegression Line
plt.plot(testing_x,lin.predict(testing_x), color = "blue")
#Title,Labels resp:
plt.title(" TESTING salary and expirence table")
plt.xlabel("Expirence")
plt.ylabel("Salary")
plt.show()

