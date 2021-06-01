import sys

import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine, MetaData, Table

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV

import pickle

""" 
The script has 7 functions. The main fn. requires 2 arguments from the command line, which are: 
database_filepath: Path necessary to find the database to load. 
model_filepath: Path in which to save the resulting model.

To run the script use this in the CLI:
python3 train_classifier.py "../data/output_etl.db" "model.pkl"
"""


def load_data(database_filepath):
    """ Load the database, and spit the data into predictors and predicted variables """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('df_clean', engine)

    # Split the dataset (target and features)
    X = df['message']  # predictors (message)
    Y = df.iloc[:,range(4,40)]  # categories to predict

    print("\n\nShape of X" , X.shape)
    print("Shape of Y" , Y.shape)

    return X, Y, Y.columns
    # pass


def tokenize(text):
    """ For a chr array: Tokenize, lemmatize, put in lowercase and split into tokens """

    # Normalization
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenizing
    tokens = word_tokenize(text)
    
    # Reduce words to their root form and remove stop words
    stop_words = stopwords.words("english")
    clean_text = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stop_words]

    return clean_text


def model_report(y_pred, y_test):
    ''' Show the accuracy of the model for each category '''

    labels = np.unique(y_pred)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels, "\n")
    print("Accuracy obtained for each Category:\n", accuracy, "\n")
    print("------------------------------------------------------")

    for i, col in enumerate(y_test):
        print("Category: {}".format(col))
        print("------------------------------------------------------")
        print(classification_report(y_test[col], y_pred[:, i], zero_division=0) )




def build_model():
    ''' Build pipeline model using different variations of tfidf and max_features of the RF '''
    
    # pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(n_estimators=5, random_state=97, n_jobs=-1, min_samples_split=200, max_features="log2", max_depth=5) ) 
    ])

    # pipeline parameters:
    parameters = {  
        'tfidf__use_idf': (True, False),
        "clf__max_features": ["log2", 0.01]
        }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ Use the model given to classify the test observations and show the results """

    y_pred = model.predict(X_test)
    model_report(y_pred, Y_test)


def save_model(model, model_filepath):
    """ Save the model into a pickle file """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ Load data from DB, build a model and train it. Then evaluate it and save the model parameters in a pkl file. """

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
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
