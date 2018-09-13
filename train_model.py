#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 22:11:13 2018

@author: Pavan Akula

References
- https://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
"""



import os
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

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

train['Family'] = (train['Parch'] + train['SibSp']).map(lambda s: 'Single' if s <= 1 else 'SmallFamily' if s >= 2 and s < 5 else 'LargeFamily' if s >= 5 else 'Unknown')



#Columns with null values
null_columns=train.columns[train.isnull().any()]
train[null_columns].isnull().sum()

train.drop('Name', inplace=True, axis=1)
train.drop('PassengerId', inplace=True, axis=1)
train.drop('Ticket', inplace=True, axis=1)
train.drop('Cabin', inplace=True, axis=1)

train['Embarked'] = train['Embarked'].fillna('C')



