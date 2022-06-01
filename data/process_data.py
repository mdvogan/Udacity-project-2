# Import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ Function to load and merge messages and categories data
    
        Args:
            messages_filepath (str): messages filepath string
            categories_filepath_filepath (str): categories filepath string
            
        Returns:
            df (dataframe): dataframe of joined messages and categories data
    """
    
    # Read raw data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets together
    df = pd.merge(messages, categories, on = ["id"])
    
    return df


def clean_data(df):
    """ Function to clean raw messages data. Two main steps:
            1) Create separate on-hot vector for each category
            2) Remove duplicates
            
        Args:
            df (dataframe): merged raw messages/categories data
        
        Returns: clean_df (dataframe): cleaned dataframe
    
    """
    # Split categories column
    categories = df["categories"].str.split(";", expand = True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # Get column names
    category_colnames = row.apply(lambda x: x[0:-2])
    categories.columns = category_colnames

    # Convert category columns into on-hot vectors
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df_clean = df.drop(columns = "categories")
    
    # concatenate the original dataframe with the new `categories` dataframe
    df_clean = pd.concat([df_clean, categories], axis = 1)
    
    # drop duplicates
    df_clean = df_clean.drop_duplicates()
    
    return df_clean


def save_data(df, database_filename):
    """ Function to save data to SQL database
    
        Args: df (dataframe): cleaned messages/categories dataframe
        database_filename (string): name of database to create/save
        
        Returns: None
    
    """
    
    # Save clean data to inputted database filename
    db_name = database_filename.split("/")[-1].replace(".db", "")
    engine = create_engine('sqlite:///{0}'.format(database_filename))
    df.to_sql(db_name, engine, index=False) 
    
    return None


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