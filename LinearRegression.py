'''
Created by Shuai Zhao, 22-Oct-2016
'''

import csv
import random
import numpy as np
import sklearn.linear_model as sklr
from sklearn.metrics import mean_squared_error
from sklearn import datasets, linear_model

with open('/home/shuai/Documents/courses/cs634/BlogFeedback/blogData_train.csv') as f:
    reader = csv.reader(f)
    my_list = list(reader)
f.closed
print("The number of observations is:", len(my_list))

#Random select 30% data
my_list2 = []
for i in range(len(my_list)):
    if random.random() < 0.3:
        my_list2.append(my_list[i])
del my_list

print("After Random selection, the total number of samples is:", len(my_list2))

#Training Data
#Turn string to float type
my_list3 = [list(map(float, my_list2[i])) for i in range(len(my_list2))]

data1 = np.array(my_list3)
X_train = data1[:, :-1]
Y_train = data1[:, -1:]
print("The number of training data is:", len(my_list3))



#Test Data
with open('/home/shuai/Documents/courses/cs634/BlogFeedback/blogData_test-2012.02.01.00_00.csv') as f:
    reader = csv.reader(f)
    test_data = list(reader)
f.closed
print("The number of test data is:", len(test_data))

test_data2 = [list(map(float, test_data[i])) for i in range(len(test_data))]
data2 = np.array(my_list3)
X_test = data1[:, :-1]
Y_test = data1[:, -1:]

#Model Training
regr = linear_model.LinearRegression()
Y_train2 = np.ravel(Y_train)
regr.fit(X_train, Y_train2)

#Model Prediction
predicted = regr.predict(X_test)
predicted1 = np.ravel(predicted)
print("The predicted values are:", predicted1)

#Calculate MSE
print("The Mean Squared Error of LR is:", mean_squared_error(Y_test, predicted1))
