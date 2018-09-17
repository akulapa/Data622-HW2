#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 20:51:58 2018

@author: Pavan Akula
"""
#import required libraries
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

#Load test data
testFile = os.path.join (os.getcwd(), 'data/test.csv')
test = pd.read_csv(testFile, delimiter = ',')

def process_ticket():
    """
    The function will process variable Ticket
    It will extract the numeric value
    One advantage of writing function after initiating is variables is they can be called using global and manipulate them
    """
    #Refer the dataset using global
    global test
    
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
    test['TicketNumber'] = test['Ticket'].map(cleanTicket)
    return test


def process_missing_age():
    #Function to calculate missing values for age
    global test
    
    #Get observations with non NA values for Age
    train_age = test.loc[(test.Age.notnull())]
    
    #Get observations with NA values for Age
    test_age = test.loc[(test.Age.isnull())]
    
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
    test.loc[(test.Age.isnull()), 'Age'] = predictedAges
    
    return test

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

test['Title'] = test['Name'].map(lambda Name:Name.split(',')[1].split('.')[0].strip())
test['Title'] = test.Title.map(Title_Dictionary)

#Extract first letter from Cabin value, store it into new column 'Deck'
test['Deck'] = test.Cabin.str[0]
test.Deck.fillna('U', inplace=True)

#Get numeric value of the ticket
test = process_ticket()

#New derived variable
test['Family'] = (test['Parch'] + test['SibSp']).map(lambda s: 'Single' if s <= 1 else 'Small' if s >= 2 and s < 5 else 'Large' if s >= 5 else 'Unknown')

#Following variable were used for deriving new variables
test.drop('Name', inplace=True, axis=1)
test.drop('PassengerId', inplace=True, axis=1)
test.drop('Ticket', inplace=True, axis=1)
test.drop('Cabin', inplace=True, axis=1)

#As fare matches the average fare impute missing value with 'C'
test['Embarked'] = test['Embarked'].fillna('C')

#Columns with null values
print("Variables with missing values")
test.isnull().sum()

#Columns with null values
#null_columns=test.columns[test.isnull().any()]
#test[null_columns].isnull().sum()

#Display missing value for fare
#Let impute missing fare value with median value
#Replace missing fare by median values
test[test['Fare'].isnull()]
fareMedian = test[((test['Fare'].notnull()) & (test['Embarked']=='S'))].Fare.median()
test['Fare'] = test['Fare'].fillna(float(fareMedian))

#Display missing value for fare
#Let impute missing Title value with based on gender
#Since gender of the passenger is female, we can impute the values 'Miss', as she is travelling by herself(SibSp and Parch is 0)
test[test['Title'].isnull()]
test['Title'] = test['Title'].fillna('Miss')

#Convet values into numeric categorical values
test['Sex'] = test['Sex'].map({'female':0, 'male':1})
test['Family'] = test['Family'].map({'Single':0, 'Small':1, 'Large':2})
test['Title'] = test['Title'].map({'Mrs':0, 'Miss':1, 'Mr':2, 'Master':3, 'Officer':4, 'Royalty':5})
test['Embarked'] = test['Embarked'].map({'C':0, 'Q':1, 'S':2})
test['Deck'] = test['Deck'].map({'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'T':7, 'U':8})

#Convert variables into categorical variables
test['Sex'] = test['Sex'].astype('category')
test['Family'] = test['Family'].astype('category')
test['Title'] = test['Title'].astype('category')
test['Embarked'] = test['Embarked'].astype('category')
test['Deck'] = test['Deck'].astype('category')
test['Pclass'] = test['Pclass'].astype('category')

#Get median value
ticketMedian = test[test['TicketNumber'].notnull()].TicketNumber.median()

#Replace missing ticketNumber by median values
test['TicketNumber'] = test['TicketNumber'].fillna(int(ticketMedian))
test['TicketNumber'] = test['TicketNumber'].astype('int')

#Impute missing ages
test = process_missing_age()

#Independent variables
X = test

# Load from file
pickleFile = os.path.join (os.getcwd(), 'data/rf_model.pkl') 
with open(pickleFile, 'rb') as file:  
    rf_model = pickle.load(file)



