import json
import plotly
import pandas as pd

import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet','stopwords'])

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and normalize case
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # display frequency of each category
    percentage = df.iloc[:,4:].sum()/len(df)
    index = df.iloc[:,4:].columns
    df_categories = list(zip(index, percentage))
    df_categories = pd.DataFrame(df_categories,columns=['Category','Percentage']).sort_values('Percentage', ascending = False)

    # display frequency of each token/word
    tokens = {}
    for message in df['message']:
        for token in tokenize(message):
            if token in tokens:
                tokens[token] += 1
            else:
                tokens[token] = 1
    df_tokens = pd.DataFrame.from_dict(tokens, orient = 'index', columns = ['Count'])
    
    # display top 30 tokens in percentage and raw count
    top30_tokens_pct = df_tokens.sort_values('Count', ascending = False)[:30]['Count']/len(df)
    top30_tokens = list(df_tokens.sort_values('Count', ascending = False)[:30].index) 
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=df_categories['Category'],
                    y=df_categories['Percentage']
                    
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "%"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 0
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top30_tokens,
                    y=top30_tokens_pct
                )
            ],

            'layout': {
                'title': 'Top 30 Most Used Words',
                'yaxis': {
                    'title': "%"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()