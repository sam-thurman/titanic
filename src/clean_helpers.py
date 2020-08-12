import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess(df):
    df.Age = df.Age.fillna(value=df.Age.mean())
    # create new class U for unkown embarking locations
    df.Embarked = df.Embarked.fillna(value='U')
    df.Embarked = df.Embarked.replace('C','Cherbourg').replace('Q','Queenstown').replace('S','Southampton')
    df.Fare = df.Fare.fillna(value=df.Fare.mean())
    df.Age = df.Age.fillna(value=df.Age.mean())
    df.set_index('PassengerId', inplace=True, drop=True)
    df.drop('Cabin', axis=1, inplace=True)
    df.drop('Ticket', axis=1, inplace=True)
    df.drop('Name', axis=1, inplace=True)
    return df

class Preprocesser:

    def __init__(self):
        pass
    
    def fit(self, X, y):
        return self

    def transform(self, X):
        X = preprocess(X)
        return X

def get_train_X_y(path_to_data_folder):
    df = pd.read_csv(f'{path_to_data_folder}/train.csv')
    df = preprocess(df)
    X = df.drop('Survived',axis=1)
    y = df.Survived
    return X, y

def get_test(path_to_data_folder):
    df = pd.read_csv(f'{path_to_data_folder}/test.csv')
    return preprocess(df)


class CustomScaler:
    '''
    This is a custom StandardScaler implementation for Pipeline.
    '''
    def __init__(self, continuous_cols=None):
        self.continuous_cols = continuous_cols
        self.ss = StandardScaler()
        print(f'creating StandardScaler object for {continuous_cols} in X') 
        pass
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.continuous = self.X[self.continuous_cols]
        self.ss.fit(self.continuous)
        return self
        
    def transform(self, X):
        self.scaled_data = self.ss.transform(self.continuous)
        self.scaled_data = pd.DataFrame(self.scaled_data, columns=self.continuous_cols)
        self.scaled_data.index = self.X.index
        self.X.drop(self.continuous_cols, axis=1, inplace=True)
        return pd.concat([self.X, self.scaled_data],axis=1, )

class CustomEncoder:
    '''
    This is a custom OneHotEncoder implementation for Pipeline
    '''
    

    def __init__(self, categorical_cols=None):
        self.categories = categorical_cols
        if categorical_cols:
            print(f'creating a OneHotEncoder object for {categorical_cols}')
        pass
    
    def fit(self, X, y):
        return self
        
        
    def transform(self, X):
        for col in self.categories:
            ohe = OneHotEncoder()
            feature = np.array(X[col]).reshape(-1,1)
            ohe.fit(feature)
            encoded = pd.DataFrame(ohe.transform(feature).toarray())
            encoded.index = X.index
            X = pd.concat([X,encoded],axis=1)
            for name in encoded.columns:
                X.rename(columns={name:f'{col}: {name}'},inplace=True)
            X.drop(col,inplace=True,axis=1)
        return X
