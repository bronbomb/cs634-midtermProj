'''
Created by Shuai Zhao, 22-Oct-2016

Data Samples:
30.83,0,1.25,1,202,0,1
58.67,4.46,3.04,6,43,560,1
?,5,8.5,0,0,0,0
52.33,1.375,9.46,0,200,100,0
?,0.5,0.835,0,320,0,0

The last column is the dependent variable. 1 means (+), 0 means (-)
'''

import csv
import numpy as np
import sklearn.linear_model as sklr
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import svm


with open('/home/shuai/Documents/courses/cs634/crx.data-v1.1.csv') as f:
    reader = csv.reader(f)
    my_list = list(reader)
f.closed
print("The number of observations is:", len(my_list))

#Delete records that have missing attributes '?'
# Use descending order to avoid the change of item index
for i in range(len(my_list) - 1, -1, -1):
    for j in range(len(my_list[1])):
        if my_list[i][j] == '?':
            del my_list[i]

print("After deleting observations containing missing values, the total number of observations is:", len(my_list))

#Split training data and test data
#Turn string to float type
my_list2 = [list(map(float, my_list[i])) for i in range(len(my_list))]

data1 = np.array(my_list2)
X = data1[:, :-1]
Y = data1[:, -1:]
Y = Y.astype(int)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
print("Training Data:", X_train.shape[0], "\nTest Data", X_test.shape[0])
#Model Training
clf = svm.SVC()
Y_train2 = np.ravel(Y_train)
clf.fit(X_train, Y_train2)

#Predict class labels for the test set
predicted = clf.predict(X_test)
predicted1 = np.ravel(predicted)
predicted2 = np.array([round(predicted1[i]) for i in range(len(predicted1))])
print("The predicted values are:", predicted2)

print("The Accuracy Score of SVM is:", metrics.accuracy_score(Y_test, predicted2))
