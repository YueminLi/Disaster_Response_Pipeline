import sys
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
import sqlite3
import pickle

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def load_data(database_filepath):
    """
    This function loads data from database_filepath
    
    Arg:
        database_filepath: the filepath of the database to be imported
        
    Returns:
        df: a pandas dataframe loaded from database filepath
        X: feture
        y: target
        category_names: labels
    """
    # create an engine
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    # read the table into a pandas dataframe
    df = pd.read_sql_table('DisasterResponse', engine)
    # define feature
    X = df.message.values
    # define target
    y = df.iloc[:, 4:]
    # define labels
    category_names = y.columns
    
    return X, y, category_names

def tokenize(text):
    """
    This function adopts the following text preprocessing steps:
        a. remove punctuation
        b. get lower case
        c. tokenize text
        d. remove stopwords
        e. lemmatize token
        
    Args:
        text: messages in pandas dataframe
        
    Returns:
        clean tokens
        
    """
    # remove punctuation and get lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # remove stopwords and lemmatize
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    This function builds machine learning pipeline
    
    Returns: 
        model: machine learning model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(SVC()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'clf__estimator__C': [0.1, 1, 10]
                 }

    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=10)
    
    return model
 

def metrics(y_pred, y_true):
    """
    This function generates machine learning model's precision, recall, f1-score,
    and accuracy scores for each of 36 categories and total mean score. 
    
    Args: 
        y_pred: predicted target
        y_true: actual target, as of y_test
        
    Returns:
        metrics: a table displaying precision, recall, f1-score, accuracy scores
        for each category and total mean score
    """
    categories = []
    precision = []
    recall = []
    f1_score = []
    accuracy = []
    
    for i,col in enumerate(y_test.columns):
        categories.append(col)
        
        metrics = precision_recall_fscore_support(y_test.iloc[:,i],y_pred[:,i],average='weighted')
        precision.append(metrics[0])
        recall.append(metrics[1])
        f1_score.append(metrics[2])
        
        accuracy.append(accuracy_score(y_true.iloc[:,i],y_pred[:,i]))
    
    metrics = pd.DataFrame(
        data = {'Precision':precision,'Recall':recall,'F1-Score':f1_score, 'Accuracy': accuracy}, 
        index = categories)
    
    mean = pd.DataFrame({'Precision': [metrics['Precision'].mean()], 
                         'Recall': [metrics['Recall'].mean()],
                        'F1-Score': [metrics['F1-Score'].mean()],
                        'Accuracy': [metrics['Accuracy'].mean()]}).rename(index={0: 'model_mean'})
    
    print(mean)
    metrics = pd.concat([mean, metrics])
    
    return metrics


def evaluate_model(model, X_test, Y_test, category_names):
    """
    evaluate the performance of the machine learning pipeline on test data
	
	Args:
		model: machine learning model
		X_test: test feature
		Y_test: test target
		category_names: labels
        
    Returns:
        metrics: a table displaying precision, recall, f1-score, accuracy scores
        for each category and total mean score
    """    
    # predict target using the machine learning model
    Y_pred = model.predict(X_test)
	
    # print metrics
    metrics = metrics(Y_test, Y_pred)
    print(metrics)

def save_model(model, model_filepath):
    """
    This function exports machine learning model as a pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
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
