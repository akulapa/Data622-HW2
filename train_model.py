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

#import required libraries
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

#Load training data
trainFile = os.path.join (os.getcwd(), 'data/train.csv')
train = pd.read_csv(trainFile, delimiter = ',')

def process_ticket():
    """
    The function will process variable Ticket
    It will extract the numeric value
    One advantage of writing function after initiating is variables is they can be called using global and manipulate them
    """
    #Refer the dataset using global
    global train
    
    # a function that extracts numeric value from the ticket, returns 'Unk' if no numeric postfix
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
            return np.nan
    
    # map cleanTicket function and extract the value for each row:
    train['TicketNumber'] = train['Ticket'].map(cleanTicket)
    return train


def process_missing_age():
    #Function to calculate missing values for age
    global train
    
    #Get observations with non NA values for Age
    train_age = train.loc[(train.Age.notnull())]
    
    #Get observations with NA values for Age
    test_age = train.loc[(train.Age.isnull())]
    
    #Seperate X and y values for training dataset
    X_train = train_age.drop('Age', axis=1).values
    y_train = train_age['Age'].values
    
    #As we will imputing missing Age values, drop the column from test dataset
    X_test = test_age.drop('Age', axis=1).values
    
    #Lets use all parameters as default, with number trees as 500 in random forest regressor
    ramdomForest = RandomForestRegressor(n_estimators=500, n_jobs=-1)
    ramdomForest.fit(X_train, y_train)
    
    #Predict the missing values
    predictedAges = ramdomForest.predict(X_test)
    
    #Apply to the dataset
    train.loc[(train.Age.isnull()), 'Age'] = predictedAges
    
    return train

#Wikipedia suggests "Jonkheer" and "Countess" is honorific titile
#https://en.wikipedia.org/wiki/Jonkheer
#https://en.wikipedia.org/wiki/Count
#Create dictionary and map them more generic values

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

#Split name as comma seperated values and get second value(1) from the list
#Split the value again based on seperator(.), get first value(0)
#ultimately get prefix value

train['Title'] = train['Name'].map(lambda Name:Name.split(',')[1].split('.')[0].strip())
train['Title'] = train.Title.map(Title_Dictionary)

#Extract first letter from Cabin value, store it into new column 'Deck'
train['Deck'] = train.Cabin.str[0]
train.Deck.fillna('U', inplace=True)

#Get numeric value of the ticket
train = process_ticket()

#New derived variable
train['Family'] = (train['Parch'] + train['SibSp']).map(lambda s: 'Single' if s <= 1 else 'Small' if s >= 2 and s < 5 else 'Large' if s >= 5 else 'Unknown')

#Columns with null values
null_columns=train.columns[train.isnull().any()]
train[null_columns].isnull().sum()

#Following variable were used for deriving new variables
train.drop('Name', inplace=True, axis=1)
train.drop('PassengerId', inplace=True, axis=1)
train.drop('Ticket', inplace=True, axis=1)
train.drop('Cabin', inplace=True, axis=1)

#As fare matches the average fare impute missing value with 'C'
train['Embarked'] = train['Embarked'].fillna('C')

#Convet values into numeric categorical values
train['Sex'] = train['Sex'].map({'female':0, 'male':1})
train['Family'] = train['Family'].map({'Single':0, 'Small':1, 'Large':2})
train['Title'] = train['Title'].map({'Mrs':0, 'Miss':1, 'Mr':2, 'Master':3, 'Officer':4, 'Royalty':5})
train['Embarked'] = train['Embarked'].map({'C':0, 'Q':1, 'S':2})
train['Deck'] = train['Deck'].map({'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'T':7, 'U':8})

#Convert variables into categorical variables
train['Sex'] = train['Sex'].astype('category')
train['Family'] = train['Family'].astype('category')
train['Title'] = train['Title'].astype('category')
train['Embarked'] = train['Embarked'].astype('category')
train['Deck'] = train['Deck'].astype('category')
train['Survived'] = train['Survived'].astype('category')
train['Pclass'] = train['Pclass'].astype('category')

#Get median value
ticketMedian = train[train['TicketNumber'].notnull()].TicketNumber.median()

#Replace missing ticketNumber by median values
train['TicketNumber'] = train['TicketNumber'].fillna(int(ticketMedian))

#Impute missing ages
train = process_missing_age()

#Seperate dependent and independent variables
X = train.drop('Survived', axis=1)
y = train['Survived']

#We will be building Random forest regression model
rf_model = RandomForestRegressor(random_state=1, n_estimators=1000, min_samples_split=2, min_samples_leaf=7, oob_score=True)
rf_model.fit(X, y)

#Save output to picket file
pickleFile = os.path.join (os.getcwd(), 'data/rf_model.pkl') 
pickle.dump(rf_model, open(pickleFile, 'wb'))


