# import libraries
import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """ Function to load cleaned messages datafor model training
    
        Args: database_filepath (string): messages database name
        
        Returns:
            X (dataframe): messages feature data
            Y (dataframe): category target labels
            category_names (list): category name list
    """
    
    # load data from database
    db_name = database_filepath.split("/")[-1].replace(".db", "")
    engine = create_engine('sqlite:///{0}'.format(database_filepath))
    df = pd.read_sql_table(db_name, engine)
    
    # Split into feature matrix (X) and target variable matrix (Y)
    X = df[["message"]]
    Y = df.drop(columns = ["id", "message", "original", "genre", "related"])
    
    # Get category names list
    category_names = list(Y.columns)
    
    return X, Y, category_names


def tokenize(text):
    """Function to tokenize test to create model features
       
       Args: 
        text (str): text to tokenize
        
       Returns:
        clean_tokens (list): clean tokens for model training
    """

    # Remove special characters
    text = re.sub(r"^a-zA-Z0-9", "", text)

    # Tokenize
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()

    # Clean tokens by removing case and whitespace
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """ Function to build machine learning model pipeline
    
        Args:
            None
            
        Returns:
            cv (GridSearchCV): GSCV model object
    """
    
    # Create model pipeline
    pipeline =  Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(estimator = RandomForestClassifier()))
    ])
    
    # Gridsearch CV over important parameters
    parameters = {
    'tfidf__smooth_idf': [True, False],
    'clf__estimator__n_estimators': [10, 20]
    }

    cv = GridSearchCV(pipeline, param_grid = parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ Function to evalaute model performance
        Prints classification report for each message category
        Args: 
            model (GridSearchCV): fitted model GSCV
            X_test (dataframe): holdout test messages
            Y_test (dataframe): holdout message labels
            category_names (list): category names
            
        Returns None
    """
    # Create test set predictions
    y_test_pred = model.predict(X_test["message"])

    # Print classification report for every category
    for i, cat in enumerate(category_names):
        print("Column: {0}".format(cat))
        print(classification_report(Y_test[cat], y_test_pred[:, i]))
    
    return None
    
def save_model(model, model_filepath):
    """ Function to save model
    
        Args:
            model (GridSearchCV): fitted model GSCV to save
            model_filepath (str): filepath to save model
            
        Returns None
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    
    return None


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train["message"], Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()