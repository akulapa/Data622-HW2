#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 22:11:13 2018
@author: Pavan Akula
References
- https://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
- https://www.datacamp.com/community/tutorials/categorical-data
- https://www.youtube.com/watch?v=0GrciaGYzV0
"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import pickle


trainFile = os.path.join (os.getcwd(), 'data/train.csv')
train = pd.read_csv(trainFile, delimiter = ',')

def process_ticket():
    
    global train
    
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip(), ticket)
        ticket = filter(lambda t : t.isdigit(), ticket)
        ticket = ''.join(ticket)
        if len(ticket) > 0:
            return ticket
        else: 
            return 'XXX'
    
    # Extracting dummy variables from tickets:
    train['TicketNumber'] = train['Ticket'].map(cleanTicket)
    return train


def process_missing_age():
    
    global train
    
    train_age = train.loc[(train.Age.notnull())]
    test_age = train.loc[(train.Age.isnull())]
    
    X_train = train_age.drop('Age', axis=1).values
    y_train = train_age['Age'].values
    
    X_test = test_age.drop('Age', axis=1).values
    
    ramdomForest = RandomForestRegressor(n_estimators=500, n_jobs=-1)
    ramdomForest.fit(X_train, y_train)
    
    predictedAges = ramdomForest.predict(X_test)
    train.loc[(train.Age.isnull()), 'Age'] = predictedAges
    
    return train


Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}
    
train['Title'] = train['Name'].map(lambda Name:Name.split(',')[1].split('.')[0].strip())
train['Title'] = train.Title.map(Title_Dictionary)

train['Deck'] = train.Cabin.str[0]
train.Deck.fillna('U', inplace=True)

train = process_ticket()

train['Family'] = (train['Parch'] + train['SibSp']).map(lambda s: 'Single' if s <= 1 else 'Small' if s >= 2 and s < 5 else 'Large' if s >= 5 else 'Unknown')



#Columns with null values
null_columns=train.columns[train.isnull().any()]
train[null_columns].isnull().sum()

train.drop('Name', inplace=True, axis=1)
train.drop('PassengerId', inplace=True, axis=1)
train.drop('Ticket', inplace=True, axis=1)
train.drop('Cabin', inplace=True, axis=1)

train['Embarked'] = train['Embarked'].fillna('C')

train['Sex'] = train['Sex'].map({'female':0, 'male':1})
train['Family'] = train['Family'].map({'Single':0, 'Small':1, 'Large':2})
train['Title'] = train['Title'].map({'Mrs':0, 'Miss':1, 'Mr':2, 'Master':3, 'Officer':4, 'Royalty':5})
train['Embarked'] = train['Embarked'].map({'C':0, 'Q':1, 'S':2})
train['Deck'] = train['Deck'].map({'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'T':7, 'U':8})

train['Sex'] = train['Sex'].astype('category')
train['Family'] = train['Family'].astype('category')
train['Title'] = train['Title'].astype('category')
train['Embarked'] = train['Embarked'].astype('category')
train['Deck'] = train['Deck'].astype('category')
train['Survived'] = train['Survived'].astype('category')
train['Pclass'] = train['Pclass'].astype('category')

train['TicketNumber'] = train['TicketNumber'].map(lambda t: train[train['TicketNumber'] != 'XXX'].TicketNumber.median() if t=='XXX' else float(t))

train = process_missing_age()

X = train.drop('Survived', axis=1)
y = train['Survived']

model = RandomForestRegressor(n_estimators=100, random_state=1, oob_score=True)
model.fit(X, y)
roc_auc_score(y, model.oob_prediction_)

features = pd.DataFrame()
features['feature'] = X.columns
features['importance'] = model.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh')

predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'Deck', 'TicketNumber', 'Family']

results = []
n_estimators_options = [10, 20, 40, 80, 150, 200, 300, 500, 600, 700, 800, 900, 1000, 1200, 1500, 1700, 1800, 2000]
for trees in n_estimators_options:
    model = RandomForestRegressor(random_state=1, n_estimators=trees, min_samples_split=2, min_samples_leaf=1, oob_score=True)
    model.fit(train[predictors], train['Survived'])
    print (trees, 'trees')
    roc = roc_auc_score(train['Survived'], model.oob_prediction_)
    print ("ROC:", roc)
    results.append(roc)
    print ("")
    
pd.Series(results, n_estimators_options).plot()

results = []
leaf_option = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for leaf in leaf_option:
    model = RandomForestRegressor(random_state=1, n_estimators=1000, min_samples_split=2, min_samples_leaf=leaf, oob_score=True)
    model.fit(train[predictors], train['Survived'])
    print (leaf, 'leaf')
    roc = roc_auc_score(train['Survived'], model.oob_prediction_)
    print ("ROC:", roc)
    results.append(roc)
    print ("")
    
pd.Series(results, leaf_option).plot()

model = RandomForestRegressor(random_state=1, n_estimators=1000, min_samples_split=2, min_samples_leaf=6, oob_score=True)
model.fit(train[predictors], train['Survived'])
roc = roc_auc_score(train['Survived'], model.oob_prediction_)
print ("ROC:", roc)

pickleFile = os.path.join (os.getcwd(), 'data/modelPickle.pkl') 
pickle.dump(model, open(pickleFile, 'wb'))


