
import sys

import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

#https://www.python.org/dev/peps/pep-0008/#:~:text=Imports%20are%20always%20put%20at,before%20module%20globals%20and%20constants.


def load_data(messages_filepath, categories_filepath):
    """ Merge two csv files by id and return a data frame """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    print(messages.shape)
    print(categories.shape)

    df = messages.merge(categories, how="inner",on="id")
    print(df.shape)

    return df


def clean_data(df):
    """Merge two csv files by id and return a data frame"""
    
    categories = df["categories"].str.split(";", expand=True)

    categ_names = df["categories"].str.split(";", expand=True).iloc[0]
    categ_names = categ_names.str.replace(  r"-.+", "", regex=True)
    zip_aux = zip(categories.columns.tolist(), categ_names.tolist())
    dict_names = dict(zip_aux)
    categories.rename(columns=dict_names, inplace=True)

    for column in categ_names:
        
        # set each value to be the last character of the string
        categories[column] = categories[column].str.replace(  r".+-", "", regex=True)
        # convert column from string to numeric
        categories[column] = categories[column].astype("int32")
    
    # drop the original categories column from `df`
    df = df.drop(columns=["categories"], axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    
    # DUPLICATES 
    # check number of duplicates
    unique_id, count = np.unique(df.id, return_counts=True)
    df_clean = df[ ~ df.duplicated(keep='first') ]


def save_data(df, database_filename):
    """Save a data frame as a database using a given path"""
    
    conn = sqlite3.connect(database_filename)
    df.to_sql('df_clean', con = conn, if_exists='replace', index=False)
    # commit any changes to the database and close the database
    conn.commit()
    conn.close()  

    #from sqlalchemy import create_engine
    #engine = create_engine('sqlite:///InsertDatabaseName.db')
    #df.to_sql('InsertTableName', engine, index=False)


def main():
    """Load two csv files, clean them and save them in a database"""

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()