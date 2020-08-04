import numpy as np
import pandas as import pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def drop_features(df, col_list=[None]):
  '''
  This function drops multiple columns from a dataframe in place

  Input: a pandas df, list of column names as strings
  '''
  # loop through column list
  for col in col_list:
    # drop current column from df
    df = df.drop(col, axis=1)

def encode_and_concat_feature(X, feature_name):
   '''
    Helper function for transforming a feature into multiple columns of 1s and 0s. Used
    in both training and testing steps. 

    Input: full X dataframe, feature name 
    Output: dataframe with given feature transformed into multiple columns of 1s and 0s
   '''
    # create new one-hot encoded df based on the feature
    ohe = OneHotEncoder()
    single_feature_df = X[[feature_name]]
    feature_array = ohe.fit_transform(single_feature_df).toarray()
    ohe_df = pd.DataFrame(feature_array, columns=ohe.categories_[0])
    # drop the old feature from X and concat the new one-hot encoded df
    X = pd.concat([X, ohe_df], axis=1)

    return X

def hot_encode_titanic(X):
  '''
  This is a catch-all function for the encoding done on the titanic dataset.  Encodes and
  renames the categorical features used in analysis

  Input: full X dataframe
  Output: full X dataframe with encoded categorical (now binary) variables
  '''
  X = encode_and_concat_feature(X,'Pclass')
  for i in [1,2,3]:
    X.rename(columns={i:f'Pclass: {i}'},inplace=True)
  X.drop('Pclass',axis=1,inplace=True)
  X = encode_and_concat_feature(X,'Sex')
  X.drop('Sex',axis=1,inplace=True)
  X = encode_and_concat_feature(X,'SibSp')
  for i in range(9):
    X.rename(columns={i:f'SibSp: {i}'},inplace=True)
  X.drop('SibSp',axis=1,inplace=True)
  X = encode_and_concat_feature(X,'Parch')
  for i in range(7):
    X.rename(columns={i:f'Parch: {i}'},inplace=True)
  X.drop('Parch',axis=1,inplace=True)
  X = encode_and_concat_feature(X,'Embarked')
  X.drop('Embarked',axis=1,inplace=True)

  return X
def rename_numerically_named_columns(df, col_list=[None]):
  '''
  This function renames ordered and numberd (0,1,2...) columns in place
  '''
  # this line zips our column list with a list of numeric values (0-n) corresponding to the 
  # names of the scaled columns
  # it's then coerced into a dictionary and fed as keys/vals to the rename method
  df = df.rename(dict(zip(col_list,list(range(len(col_list))))),axis=1)

def scale_numeric_features(X, col_list=[None]):
  '''
  This function uses sklearn to scale numeric features of our dataframe
  Input: full X dataframe (indicators)
  Output: full X dataframe with scaled/renamed numeric features
  '''
  numeric_features = X[col_list]
  ss = StandardScaler()
  numeric_features = ss.fit_transform(numeric_features)
  X = pd.concat([pd.DataFrame(numeric_features),X.drop(col_list,axis=1)],axis=1)
  rename_scaled_columns(numeric_features,col_list=numeric_features)
  return X

def get_data(path_to_csv):
  '''
  This function retrieves and cleans all of the data we need for analysis
  on the Titanic dataset
  Input: path to csv containing Titanic data
  Output: df, X, y, X_train, X_test, y_train, y_test
  '''
  df = pd.read_csv(path_to_csv)
  df.Age=df.Age.fillna(value=df.Age.mean())
  df.Embarked=df.Embarked.fillna(value='Missing')
  df.Embarked=df.Embarked.replace('C','Cherbourg').replace('Q','Queenstown').replace('S','Southampton')
  drop_features(df,col_list=['PassengerId','Cabin','Ticket','Name'])
  df = df.dropna()
  y = df.Survived
  X = df.drop('Survived',axis=1)
  X = hot_encode_titanic(X)
  X = scale_numeric_features(X,col_list=['Age','Fare'])
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,statify=y)
  return df, X, y, X_train, X_test, y_train, y_test