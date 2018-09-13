#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 22:11:13 2018

@author: Pavan Akula

References
- https://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
- https://www.datacamp.com/community/tutorials/categorical-data
"""



import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.model_selection import cross_val_predict
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

#train['Sex'] = train['Sex'].astype('category')
#train['Family'] = train['Family'].astype('category')
#train['Title'] = train['Title'].astype('category')
#train['Embarked'] = train['Embarked'].astype('category')
#train['Deck'] = train['Deck'].astype('category')
#train['Survived'] = train['Survived'].astype('category')
#train['Pclass'] = train['Pclass'].astype('category')


train['TicketNumber'] = train['TicketNumber'].map(lambda t: train[train['TicketNumber'] != 'XXX'].TicketNumber.median() if t=='XXX' else float(t))

train = process_missing_age()

predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'Deck', 'TicketNumber', 'Family']
lr = LogisticRegression(random_state=1)
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)

scores = cross_val_score(lr, train[predictors], train['Survived'], scoring='f1', cv=cv)

print(scores.mean())

rf = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)

kf = KFold(train.shape[0], n_folds=5, random_state=1)
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)

predictions = cross_validation.cross_val_predict(rf, train[predictors], train['Survived'], cv=kf)

predictions = pd.Series(predictions)

scores = cross_val_score(rf, train[predictors], train['Survived'], scoring='f1', cv=kf)

print(scores.mean())
