import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # Load relevant files
    messages = pd.read_csv("disaster_messages.csv")
    categories = pd.read_csv("disaster_categories.csv")
    
    return messages, categories


def clean_data(messages, categories):
    # Merge messages and categories together
    df = messages.merge(categories, on="id")
    
    # Separate categories single column into separate columns
    categories = df['categories'].str.split(";", expand = True)
    
    # Get first row
    row = categories.iloc[0]
    
    # Clean first row to get column names and apply
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    # Clean each category column
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = pd.to_numeric(categories[column])
    
    # Drop single categories column
    df.drop(columns = ['categories'], axis=1, inplace = True)
    
    # Merge on new separated categories
    df = df.merge(categories, left_index = True, right_index = True)
    
    # Remove duplicate rows
    df = df[~df.duplicated()]
    
    return df


def save_data(df, database_filename):
    # Create sql database
    engine = create_engine('sqlite:///' + database_filename)
    # Send df to database
    df.to_sql('database', engine, index=False)
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)
        
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