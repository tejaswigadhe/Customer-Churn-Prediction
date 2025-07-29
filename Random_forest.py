#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv('Churn_dataset.csv')

# Drop 'ID' and 'DOB' columns
columns = ['ID', 'DOB']
df.drop(columns, inplace=True, axis=1)

# Map categorical values to numeric
mymap = {'Yes': 1, 'No': 0, 'NO': 0, ' ': 0, 'Low': 1, 'Medium': 2, 'High': 3}
df = df.applymap(lambda s: mymap.get(s) if s in mymap else s)

# Rename columns
df = df.rename(index=str, columns={'LastSatisfactionSurveyScore(1=V.Poor, 5=V.High)': 'LastSatisfactionSurveyScore'})

# Fill missing values for categorical columns
df['Occupation'].fillna('IT', inplace=True)
df['Gender'].fillna('M', inplace=True)
df['IncomeLevel'].fillna(3, inplace=True)

# Create dummy variables for categorical columns
df = pd.get_dummies(df, columns=["Plan", "Occupation"], prefix=["Plan", "Occupation"])
df = pd.get_dummies(df, columns=["Gender", "SubscriberJoinDate"], prefix=["Gender", "Join_date"])

# Features and target variable
features_1 = ['LastMonthlyBill', 'LastpaidAmount', 'IncomeLevel', 'LastSatisfactionSurveyScore', 
              'Plan_A', 'Plan_B', 'Plan_C', 'Plan_D', 'Plan_E', 'Plan_F', 
              'Occupation_Business', 'Occupation_Engineer', 'Occupation_Finance', 
              'Occupation_Government', 'Occupation_IT', 'Occupation_Medical', 
              'Occupation_Others', 'Occupation_Sales', 'Gender_F', 'Gender_M', 
              'Join_date_15/1/17', 'Join_date_21/3/16', 'Join_date_26/5/15', 
              'Join_date_3/10/13', 'Join_date_30/7/14']
X = df[features_1].values
y = df['Churn'].values

# Handling missing values (Impute with mean for numerical data)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 0:2] = imputer.fit_transform(X[:, 0:2])

# Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print(accuracies.mean())
print(accuracies.std())

