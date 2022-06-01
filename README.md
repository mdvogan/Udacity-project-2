# Udacity-project-2

This repository is for the second project of the Udacity Data Science nanodegree.
The goal of the project was to train a model that classifies disaster response
messages into one of 35 categories and deploy the tool in a flask app

##Libraries used
* numpy
* pandas
* sqlalchemy
* pickle
* re
* nltk
* sklearn

## Step 1: Data ETL process
The first step to the project is to load the raw data and clean in an ETL process.

There are two raw data sources:
* data/disaster_messages.csv
* data/disaster_categories.csv

The file "data/process_data.py" reads merges the raw data, creates message category
one-hot vectors, and drops duplicates. The output of this file is a SQL table
that is then read in to "models/train_classifier.py"

## Step 2: Train classifier
The second step to this project is to train the classifier in "models/train_classifier.py"

First the data is loaded and split into a feature matrix of messages (X) and
a target matrix of message labels (Y). This data is split into an 80% training set
and 20% test set.

Next, the training data is passed through the model pipeline to train the model.
The pipeline inputs the messages, normalizes and lemmatizes the messages,
creates count vectors and applies TFIDF transformations, then fits a  multi-class
random forest classifier. The MCRF classifier is optimized using GridSearchCV
on the entire pipeline over the smooth IDF and n_estimators hyperparameters. More
could have been done on this section but adding more hyperparameters greatly
increased the run time.

Finally, the model is used to predict the test set labels and a classification
report is printed for each of the 35 categories. The model is pickled for upload
to the Flask app.

## Step 3: Flask app
The app folder contains code for a flask app that uses the trained model to
interactively classify messages. Follow the instructions below to process the data,
train the model, and run the flask app.


## Instructions to run
- To run ETL pipeline that cleans data and stores in database
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- To run ML pipeline that trains classifier and saves
    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`


## Acknowledgements
Figure Eight for providing the dataset
Udacity for providing the course and assignment review
