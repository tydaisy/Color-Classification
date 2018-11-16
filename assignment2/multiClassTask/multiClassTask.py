import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score, classification_report
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
print("B.Wavelength: 580-621")
print("C.Wavelength: 650-690")
response2 = input("please input 'A' , 'B' or 'C': ")
print ('\n')

print("What kind of regression do you prefer?")
print("A.Logistic Regression")
print("B.Neural Network")
print("C.Support Vector Machine")
response3 = input("please input 'A' , 'B' or 'C': ")
print ('\n')

#Load a matrix from csv_file
inputs = pd.read_csv("../dataset/multiclass/X.csv", header=None)
outputs = pd.read_csv("../dataset/multiclass/y.csv", header=None)
XtoClassify = pd.read_csv("../dataset/multiclass/XtoClassify.csv", header=None)
inputs = inputs.values
XtoClassify = XtoClassify.values

# Split the data into training/testing sets
if response2 == 'A':
    percentage = 0.3
if response2 == 'B':
    percentage = 0.7
if response2 == 'C':
    percentage = 0.9

# Train model
x_train, x_validation, y_train, y_validation = train_test_split(inputs, outputs, train_size = percentage, random_state=None)
y_train = np.ravel(y_train)
y_validation = np.ravel(y_validation)

# plot training dataset
blue = []
green = []
pink = []
red = []
yellow = []
print ('y_train:', y_train)
for index,number in enumerate(y_train):
    if number == 0:
        blue.append(x_train[index]) # optical reflectance intensity of various wavelengths of green color
    elif number == 1:
        green.append(x_train[index]) #  optical reflectance intensity of various wavelengths of red color
    elif number == 2:
        pink.append(x_train[index])
    elif number == 3:
        red.append(x_train[index])
    else:
        yellow.append(x_train[index])

x = np.array(pd.read_csv("../dataset/multiclass/Wavelength.csv", header=None)) # load wavelengths as x-axis
fig = plt.figure()
fig.suptitle('Dataset Visualisation', fontsize=20)

for sample in blue:
    plt.plot(x, sample, c='blue', linestyle="", marker="o", markersize=0.2)
for sample in green:
    plt.plot(x, sample, c='green', linestyle="", marker="o", markersize=0.2)
for sample in pink:
    plt.plot(x, sample, c='pink', linestyle="", marker="o", markersize=0.2)
for sample in red:
    plt.plot(x, sample, c='red', linestyle="", marker="o", markersize=0.2)
for sample in yellow:
    plt.plot(x, sample, c='yellow', linestyle="", marker="o", markersize=0.2)

plt.xlabel('Wavelength', fontsize=15)
plt.ylabel('Reflectance Intensity', fontsize=15)
plt.show()

# Extract inputs
if response1=='A':
    x_train = x_train # inputs: all features
    x_validation = x_validation
    XtoClassify = XtoClassify
if response1=='B':
    x_train = x_train[:,432:547] # filtered features (Wavelength: 580-621)
    x_validation = x_validation[:,432:547]
    XtoClassify = XtoClassify[:,432:547]
if response1=='C':
    x_train = x_train[:,629:744] # filtered features (Wavelength: 650-690)
    x_validation = x_validation[:,629:744]
    XtoClassify = XtoClassify[:,629:744]

# Data normalisation(0-1)
scaler = preprocessing.MinMaxScaler()
x_validation = scaler.fit_transform(x_validation)
x_train = scaler.fit_transform(x_train)
XtoClassify = scaler.fit_transform(XtoClassify)

# Train model
if response3=='A':
    model = linear_model.LogisticRegression()
if response3=='B':
    model = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5), random_state=1)
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
target_names = ['blue', 'green', 'pink', 'red', 'yellow']
Classification_report = classification_report(y_validation, y_pred,target_names=target_names)
print ('classification report\n', Classification_report)

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
