import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


#user interface
print("What percentage dataset you prefer to use as training set?")
print("A.30%")
print("B.70%")
print("C.90%")
response1 = input("please input 'A' , 'B' or 'C': ")
print ('\n')

print("How many features would you like to select")
print("A.all features")
print("B.Wavelength: 420-500")
print("C.Wavelength: 500-690")
print("D.Wavelength: 690-751")
response2 = input("please input 'A' , 'B' , 'C' or 'D': ")
print ('\n')

print("What kind of regression do you prefer?")
print("A.Logistic Regression")
print("B.Neural Network")
print("C.Support Vector Machine")
response3 = input("please input 'A' , 'B' or 'C': ")
print ('\n')



#Load a matrix from csv_file
inputs = pd.read_csv("../dataset/binary/X.csv", header=None)
outputs = pd.read_csv("../dataset/binary/y.csv", header=None)
XtoClassify = pd.read_csv("../dataset/binary/XToClassify.csv", header=None)
inputs = inputs.values
XtoClassify = XtoClassify.values

# Split the data into training/testing sets based on response1
if response1 == 'A':
    percentage = 0.3
if response1 == 'B':
    percentage = 0.7
if response1 == 'C':
    percentage = 0.9

x_train, x_validation, y_train, y_validation = train_test_split(inputs, outputs, train_size = percentage, random_state=None)
y_train = np.ravel(y_train)
y_validation = np.ravel(y_validation)

# plot training dataset, data Visualisation
green = []
red = []
for index,number in enumerate(y_train):
    if number == 0:
        green.append(x_train[index]) # optical reflectance intensity of various wavelengths of green color
    else:
        red.append(x_train[index]) #  optical reflectance intensity of various wavelengths of red color

x = np.array(pd.read_csv("../dataset/binary/Wavelength.csv", header=None)) # load wavelengths as x-axis
fig = plt.figure()
fig.suptitle('Dataset Visualisation', fontsize=20)
for sample in green:
    plt.plot(x, sample, c='green', linestyle="", marker="o", markersize=0.2)
for sample in red:
    plt.plot(x, sample, c='red', linestyle="", marker="o", markersize=0.2)

plt.xlabel('Wavelength', fontsize=15)
plt.ylabel('Reflective Intensity', fontsize=15)
plt.show()

# Extract inputs
if response2=='A':
    x_train = x_train # inputs: all features
    x_validation = x_validation
    XtoClassify = XtoClassify
if response2=='B':
    x_train = x_train[:,1:133] # filtered features (Wavelength: 420-470)
    x_validation = x_validation[:,1:133]
    XtoClassify = XtoClassify[:,1:133]
if response2=='C':
    x_train = x_train[:,214:744] # filtered features (Wavelength: 500-690)
    x_validation = x_validation[:,214:744]
    XtoClassify = XtoClassify[:,214:744]
if response2=='D':
    x_train = x_train[:,744:921] # filtered features (Wavelength: 690-751)
    x_validation = x_validation[:,744:921]
    XtoClassify = XtoClassify[:,744:921]

# Data normalisation(0-1)
scaler = preprocessing.MinMaxScaler()
x_validation = scaler.fit_transform(x_validation)
x_train = scaler.fit_transform(x_train)
XtoClassify = scaler.fit_transform(XtoClassify)

# Train model
if response3=='A':
    model = linear_model.LogisticRegression()
if response3=='B':
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5), random_state=None)
if response3=='C':
    model = SVC()

start_time = time.process_time()
model.fit(x_train, y_train)
end_time = time.process_time()

print ('%.9f seconds' % (end_time - start_time))

# Use validation dataset to test the accuracy of the current model
y_pred = model.predict(x_validation)
accuracy = accuracy_score(y_validation, y_pred)
print ('accuracy:', accuracy)
Confusion_matrix = confusion_matrix(y_validation, y_pred)
print ('confusion matrix', Confusion_matrix)
# Use the current trained model to predict the output of the data from file 'PredictedClasses.csv'
YtoClassify = model.predict(XtoClassify)

# Store the predictions to the file 'PredictedClasses.csv'
if response3=='A':
    dataframe = pd.DataFrame({'Logistic Regression':YtoClassify})
if response3=='B':
    dataframe = pd.DataFrame({'Neural Network':YtoClassify})
if response3=='C':
    dataframe = pd.DataFrame({'Support Vector Machine':YtoClassify})
dataframe.to_csv("PredictedClasses.csv",index=False,sep=',')
