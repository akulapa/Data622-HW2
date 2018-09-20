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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib


class derivedColumns(BaseEstimator, TransformerMixin):
    """
    This function creates derived columns using existing columns.
    """
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, *_):
        return self
    
    def transform(self, X, *_):
        
        df = X.copy()
        
        def cleanTicket(ticket):
            #Extract numeric values from ticket
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
            
        #Based on column extract derived columns
        if self.columns is not None:
            for column in self.columns:
                
                if column == 'Name':
                    #Wikipedia suggests "Jonkheer" and "Countess" is honorific titile
                    #https://en.wikipedia.org/wiki/Jonkheer
                    #https://en.wikipedia.org/wiki/Count
                    #Create dictionary and map them more generic values

                    Title_Dictionary = {
                        "Capt": "Officer", "Col": "Officer", "Major": "Officer", "Jonkheer": "Royalty",
                        "Don": "Royalty", "Sir" : "Royalty", "Dr": "Officer", "Rev": "Officer",
                        "the Countess":"Royalty", "Mme": "Mrs", "Mlle": "Miss", "Ms": "Mrs",
                        "Mr" : "Mr", "Mrs" : "Mrs", "Miss" : "Miss", "Master" : "Master", "Lady" : "Royalty"
                    }

                    #Split name as comma seperated values and get second value(1) from the list
                    #Split the value again based on seperator(.), get first value(0)
                    #ultimately get prefix value
                    df['Title'] = df['Name'].map(lambda Name:Name.split(',')[1].split('.')[0].strip())
                    df['Title'] = df.Title.map(Title_Dictionary)

                if column == 'Ticket':
                    # map cleanTicket function and extract the value for each row:
                    df['TicketNumber'] = df['Ticket'].map(cleanTicket)

                if column == 'Family':
                    #Get number of members in the family
                    #Get family size
                    df['Family'] = (df['Parch'] + df['SibSp']).map(lambda s: 'Single' if s <= 1 else 'Small' if s >= 2 and s < 5 else 'Large' if s >= 5 else 'Unknown')
                    
                if column == 'Deck':
                    #Extract first letter from Cabin value, store it into new column 'Deck'
                    df['Deck'] = df.Cabin.str[0]
        return df
    
class eliminateColumns(BaseEstimator, TransformerMixin):
    """
    This function removed columns from dataset.
    """
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, *_):
        return self
    
    def transform(self, X, *_):
        
        df = X.copy()
        
        if self.columns is not None:
            #Drop the variables
            df.drop(self.columns, inplace=True, axis=1)
       
        return df

class imputeColumns(BaseEstimator, TransformerMixin):
    """
    This function fills in missing values in the dataset.
    """
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, *_):
        return self
    
    def transform(self, X, *_):
        
        #Copy train dataset, make transformations on df
        df = X.copy()
        
        def getTitle(gender):
            #Impute missing title based on gender
            title = ''
            if gender == 'female':
                title = 'Miss'
            else:
                title = 'Mr'
            return title
            
        
        if self.columns is not None:
            for column in self.columns:
                #Impute title, get null values and apply function getTitle using lambda
                if column == 'Title':
                    dfTitle = df[df['Title'].isnull()]
                    
                    title = dfTitle.apply(lambda row: getTitle(row['Sex']), axis=1)
                    df.loc[(df.Title.isnull()), 'Title'] = title
                    
                if column == 'Fare':
                    #Get median fare based on boarding port
                    fareMedian = df[((df['Fare'].notnull()) & (df['Embarked']=='S'))].Fare.median()
                    df['Fare'] = df['Fare'].fillna(float(fareMedian))
                
                if column == 'Deck':
                    #Since there more observations make missing Deck into its own category
                    df['Deck'] = df['Deck'].fillna('U')
                
                if column == 'TicketNumber':
                    #Get median value
                    #Replace missing ticketNumber by median values
                    ticketMedian = df[df['TicketNumber'].notnull()].TicketNumber.median()
                    df['TicketNumber'] = df['TicketNumber'].fillna(int(ticketMedian))
                    
                if column == 'Embarked':
                    #As fare matches the average fare impute missing value with 'C'
                    df['Embarked'] = df['Embarked'].fillna('C')
                if column == 'Age':

                    #Convet values into numeric categorical values
                    df['Sex'] = df['Sex'].map({'female':0, 'male':1})
                    df['Family'] = df['Family'].map({'Single':0, 'Small':1, 'Large':2})
                    df['Title'] = df['Title'].map({'Mrs':0, 'Miss':1, 'Mr':2, 'Master':3, 'Officer':4, 'Royalty':5})
                    df['Embarked'] = df['Embarked'].map({'C':0, 'Q':1, 'S':2})
                    df['Deck'] = df['Deck'].map({'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'T':7, 'U':8})

                    #Convert variables into categorical variables
                    df['Sex'] = df['Sex'].astype('category')
                    df['Family'] = df['Family'].astype('category')
                    df['Title'] = df['Title'].astype('category')
                    df['Embarked'] = df['Embarked'].astype('category')
                    df['Deck'] = df['Deck'].astype('category')
                    df['Pclass'] = df['Pclass'].astype('category')

                    #Get observations with non NA values for Age
                    train_age = df.loc[(df.Age.notnull())]

                    #Get observations with NA values for Age
                    test_age = df.loc[(df.Age.isnull())]

                    #Seperate X and y values for training dataset
                    X_age_train = train_age.drop('Age', axis=1).values
                    y_age_train = train_age['Age'].values

                    #As we will imputing missing Age values, drop the column from test dataset
                    X_age_test = test_age.drop('Age', axis=1).values

                    #Lets use all parameters as default, with number trees as 500 in random forest regressor
                    ramdomForest = RandomForestRegressor(n_estimators=500, n_jobs=-1)
                    ramdomForest.fit(X_age_train, y_age_train)

                    #Predict the missing values
                    predictedAges = ramdomForest.predict(X_age_test)

                    #Apply to the dataset
                    df.loc[(df.Age.isnull()), 'Age'] = predictedAges


        return df

#Load training data
testFile = os.path.join (os.getcwd(), 'data/test.csv')
test = pd.read_csv(testFile, delimiter = ',')

#New variables are derived using following
derived_cols = ['Name', 'Ticket', 'Family', 'Deck']
#Remove not used columns
elmi_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
#Impute the columns and convert them into categorical values
imp_cols = ['Title', 'Fare', 'Deck', 'TicketNumber','Embarked','Age']

#Independent variables
X = test

# Load from file
from sklearn.externals import joblib
pickleFile = os.path.join (os.getcwd(), 'data/rf_model.pkl') 
a = joblib.load(pickleFile)
b = a.predict(X)
