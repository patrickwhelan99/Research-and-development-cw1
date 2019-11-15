# -*- coding: utf-8 -*-

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

from ann_visualizer.visualize import ann_viz

classifier = Sequential() # Initialising the ANN

classifier = Sequential() # Initialising the ANN
classifier.add(Dense(units = 4, activation = 'relu', input_dim = 30))
classifier.add(Dense(units = 10, activation = 'relu'))
classifier.add(Dense(units = 8, activation = 'relu'))
classifier.add(Dense(units = 6, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy')


classifier.fit(X_train, Y_train, batch_size = 1, epochs = 100)

Y_pred = classifier.predict(X_test)
Y_pred = [ 1 if y>=0.5 else 0 for y in Y_pred ]


total = 0
correct = 0
wrong = 0
truePositive = 0
trueNegative = 0
falseNegative = 0
falsePositive = 0

for i in range(len(Y_pred)):
  total=total+1
  if(Y_test.at[i,0] == Y_pred[i]):
    correct=correct+1
    if(Y_test.at[i,0] == 0 and Y_pred[i] == 0):
      trueNegative = trueNegative + 1
    elif(Y_test.at[i,0] == 1 and Y_pred[i] == 1):
      truePositive = truePositive + 1

  else:
    wrong=wrong+1
    if(Y_test.at[i,0] == 1 and Y_pred[i] == 0):
      falseNegative = falseNegative + 1
    elif(Y_test.at[i,0] == 0 and Y_pred[i] == 1):
      falsePositive = falsePositive + 1
    

print("Total " + str(total))
print("Correct " + str(correct))
print("Wrong " + str(wrong))
print("False Negatives " + str(falseNegative))
print("False Positives " + str(falsePositive))
print("True Negatives " + str(trueNegative))
print("True Positives " + str(truePositive))



ann_viz(classifier, filename="graph.png", title="6 Layer deep neural network")
