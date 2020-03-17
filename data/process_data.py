import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
	This function loads two csv files and merges them together to a pandas dataframe
	
	Args:
		messages_filepath: full or relative path to the messages file
		categories_filepath: full or relative path to the categories file
	
	Returns:
        df: a merged pandas dataframe containing messages and categories		
	"""
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on='id')
    
    return df
    

def clean_data(df):
    """
    This function splits one uniform category column in to 36 categories, assigns 
    its value to be 0 or 1, and removes duplicates in the pandas dataframe
    
    Args: 
        df: a raw pandas dataframe awaiting cleaning
        
    Returns:
        df: a cleaned pandas dataframe containing messages and 36 categories
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.loc[0]

    # extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # convert category values to numbers 0 or 1
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # replace 'related' column value equals to 2 with 1 
    # category values should be 0 or 1
    categories['related'].replace(2, 1, inplace=True)
    # drop child_alone column as it only has 0 value
    categories.drop(['child_alone'], axis=1, inplace=True)
        
    # drop the original categories column from `df`
    df = df.drop(['categories'],axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],axis=1,join='inner')
    
    # drop duplicates
    df = df.drop_duplicates()

    return df
    

def save_data(df, database_filename):
    """
    This function saves the cleaned data set as a sqlite database
    
    Args:
        df: the pandas dataframe to be saved
		database_filename: the name for the sqlite database file	
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponse', engine, index=False)


def main():
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
