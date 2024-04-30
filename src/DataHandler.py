#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Will make classes for more modular and reusable code
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
class DataHandler:
    """
    Responsible for loading, preprocessing, and splitting data:
        filepath (str): The path to the dataset file
        columns_to_drop (list): Columns to be removed from the dataset
        target (str): The target variable for prediction
        dummy_variables (bool): Flag to decide if categorical variables should be converted to dummy/indicator variables
    """
    def __init__(self, filepath, drop_columns, target):
        # Initializes the DataHandler with file path, columns to drop, and target column
        self.filepath = filepath
        self.drop_columns = drop_columns
        self.target = target
        self.sampled_df = None
    def load_and_prepare(self):
        # Loads and preprocesses the dataset by dropping specified columns and encoding categorical variables if specified
        df = pd.read_csv(self.filepath, sep=';')
        # Dropping 'duration' column for leakage, and rows where it was 0 since those are automatic no's (bad data)
        df = df[df['duration'] != 0]
        df = df.drop(self.drop_columns, axis=1)
        df = pd.get_dummies(df, drop_first=True) # Encodes nominal -> binary
        if len(df)>20000:
            df = df.sample(n=20000, random_state=42) # Limits length of data for training time
        self.features = df.drop(self.target, axis=1)
        self.labels = df[self.target]
        # Returns the features and labels of the dataset
        return self.features, self.labels

    def split_data(self, test_size=0.2, random_state=42):
        # Splits the data into training and testing sets, scales
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=test_size, random_state=random_state)
        scaler = StandardScaler()
        X_train_scaled =scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

