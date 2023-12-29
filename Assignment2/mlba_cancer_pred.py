'''
MLBA Assignment 2
Created by Group 66
Members : Akhil P Dominic, Rajith Ramachandran, Sarvani Gupta

Cancer prediction
'''

#Downloading the given packages
import subprocess
packages_to_install = ['pandas', 'numpy','sklearn','imbalanced-learn']
for package in packages_to_install:
  subprocess.check_call(['pip', 'install', package])
  
#importing the necessary packages
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
import csv
#Parsing the data from command line
parser=argparse.ArgumentParser()
parser.add_argument("--files",nargs='+',help="Train and test files")
args = parser.parse_args()
args=args.files
train_csv=args[0]
test_csv=args[1]
#reading the training csv file
train_data = pd.read_csv(train_csv)

#Dropping the labels from the train_data as we dont need them for training
X_train = train_data.drop("Labels", axis=1)
#Taking in the labels from the train_data
labels = train_data["Labels"]


scaler = StandardScaler()
#Applying scaling to the training dataset
X_train = scaler.fit_transform(X_train)

#Using recursive feature elimination technique to find the features which are less important and remove them so that we dont have to consider them while training
recursive_fe = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=25)
X_train = recursive_fe.fit_transform(X_train, labels)

#Splitting the dataset into training and testing dataset in the ratio 80:20
X_train, X_val, y_train, y_val = train_test_split(X_train, labels, test_size=0.2, random_state=50)

#declaring the random forest classifier with 100 estimators
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

#predicting the values from the testing dataset inorder to test AUC accuracy
y_pred = rf_classifier.predict_proba(X_val)[:, 1]

auc_score = roc_auc_score(y_val, y_pred)
print(f"Accuracy: {auc_score}")

#Importing the testing dataset
test_data = pd.read_csv(test_csv)

#Taking in and removing the ID column from testing dataset
test_ID = test_data['ID']
test_data = test_data.drop("ID", axis=1)
test_data = scaler.transform(test_data)

test_data = recursive_fe.transform(test_data)

#Predicting the final output probability
final_pred =rf_classifier.predict_proba(test_data)[:, 1]

#Writing the output predictions onto the file ./output_rf.csv
with open('./output_rf.csv', mode='w', newline='') as file:
  writer = csv.writer(file)
  writer.writerow(['ID', 'Labels'])
  for i in range(0,len(test_ID)):
    writer.writerow([test_ID[i],final_pred[i]])