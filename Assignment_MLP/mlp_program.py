# -*- coding: utf-8 -*-
"""mlp_program.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QJNvQIrmIVgqJLUVr5fQtUNLjZ18H79s
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import array,random

df = pd.read_csv("/content/data_banknote_authentication.csv")
print("No. of Rows " + str(len(df)))
df.head(10)

"""#### SPLITTING DATA INTO TRAIN AND TEST"""

train,test = train_test_split(df, test_size=0.2, random_state = 41)
N = len(train.columns) - 1
print(N)

dfs = np.split(train, np.arange(N, len(df.columns), N), axis=1)
x_train = dfs[0]
y_train = dfs[1]

dfp = np.split(test, np.arange(N, len(df.columns), N), axis=1)
x_test = dfp[0]
y_test = dfp[1]

epochs = 1500 
alpha = 0.01
weights = [1.00]     #weights[0]=bias 
dimensions =len(x_train.columns)
for i in range (1,dimensions+1):
  weights.append(random.random())
print("Initial weights are : " + str(weights))

def predict(MyList = [] , *args):
  sum = weights[0]
  for i in range(1,dimensions+1):
    sum = sum + weights[i]*MyList[i-1]
  if(sum>=0):
    return 1
  else:
    return 0


for e in range(epochs):
  for i in range(len(x_train)):
    X = np.array(x_train.values[i])
    Y = np.array(y_train.values[i])
    prediction = predict(X)
    error = Y - prediction
    weights[0] = weights[0] + alpha*error
    for d in range(1,dimensions+1):
      weights[d] = weights[d] + (alpha*error*X[d-1])
#Learning

Weights = []
for weight in weights:
  Weights.append(weight[0])
weights = Weights
print("Final weights are: "+ str(weights))

"""### FOR TRAINING DATA"""

Result_train = []
for i in range(0,len(x_train)):
  XTest = np.array(x_train.values[i])
  #print(str(XTest) + " : " + str(predict(XTest)))
  #print(predict(XTest))
  Result_train.append(predict(XTest))

print(Result_train)
FN_train = 0
TP_train = 0
TN_train = 0
FP_train = 0
for i in range(len(x_train)):
  Z = np.array(y_train.values[i])
  if(Result_train[i] == Z == 1):
    TP_train+=1
  elif(Result_train[i] == 1 and Z == 0):
    FP_train+=1
  elif(Result_train[i] == Z == 0):
    TN_train+=1
  else:
    FN_train+=1

#displaying confision matrix

ConfusionMatrix_Train = np.array([[TN_train,FP_train],[FN_train,TP_train]])
print("Confusion matrix is : \n TN FP \n FN TP")
print(ConfusionMatrix_Train)

#displaying metrics
Accuracy_train = (TP_train + TN_train)/len(x_train)
#print("Accuracy is: " + str(Accuracy*100) + "%")
Precision_train = TP_train/(TP_train+FP_train)
Recall_train = TP_train/(TP_train + FN_train)
FMeasure_train = (2*Precision_train*Recall_train)/(Precision_train+Recall_train)
print("Accuracy is: " + str(Accuracy_train)  + "\n"
+ "Precison is: "+str(Precision_train)+"\nRecall is: "+ str(Recall_train)+"\nFmeasure is: " + str(FMeasure_train))

"""### TESTING DATA"""

Result = []
for i in range(0,len(x_test)):
  XTest = np.array(x_test.values[i])
  #print(str(XTest) + " : " + str(predict(XTest)))
  Result.append(predict(XTest))

print(Result)
FN = 0
TP = 0
TN = 0
FP = 0
for i in range(len(x_test)):
  Z = np.array(y_test.values[i])
  if(Result[i] == Z == 1):
    TP+=1
  elif(Result[i] == 1 and Z == 0):
    FP+=1
  elif(Result[i] == Z == 0):
    TN+=1
  else:
    FN+=1

ConfusionMatrix = np.array([[TN,FP],[FN,TP]])
print("Confusion matrix is : \n TN FP \n FN TP")
print(ConfusionMatrix)

Accuracy = (TP + TN)/len(x_test)
#print("Accuracy is: " + str(Accuracy*100) + "%")
Precision = TP/(TP+FP)
Recall = TP/(TP + FN)
FMeasure = (2*Precision*Recall)/(Precision+Recall)
print("Accuracy is: " + str(Accuracy)  + "\n"
+ "Precison is: "+str(Precision)+"\nRecall is: "+ str(Recall)+"\nFmeasure is: " + str(FMeasure))

"""### VISUALISING CONFUSION MATRIX"""

predicted_df = pd.DataFrame({"Predicted":Result})

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predicted_df)
print(cm)
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']

plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()

"""### VISUALISING ROC CURVE"""

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted_df, pos_label=0)
# Print ROC curve
plt.plot(tpr,fpr)
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.title("True Positive Rate Vs False Positive Rate")
plt.show()


# Print AUC
auc = np.trapz(fpr,tpr)
print('AUC:', auc)

