import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
from sqlalchemy import create_engine
import joblib


""" 
The script has 4 functions. The main fn. requires 2 arguments from the command line, which are: 
database_filepath: Path necessary to find the database to load. 
model_filepath: Path in which to save the resulting model.

Be careful with filepath of the DB, the model, and the host and ports of the web app. 
"""


app = Flask(__name__)

def tokenize(text):
    """For a chr array: Tokenize, lemmatize, put in lowercase and split into tokens"""

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///Disaster_ETL.db')
#engine = create_engine('sqlite:///../data/output_etl.db')
df = pd.read_sql_table('messages', engine)
#df = pd.read_sql_table('df_clean', engine)

# load model
model = joblib.load("model.pkl")
#model = joblib.load("../models/fitted_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """Make two plots using data from the dataframe df"""

    # extract data needed for visuals
    # Barplot: types of genre of the messages
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Show distribution of different category
    categories_names = list(df.columns[4:])
    categories_counts = []
    for column_name in categories_names:
        categories_counts.append(np.sum(df[column_name]))

    
    # create visuals
    # Three charts: Genres, distribution of messages by categories, histogram of messages
    graphs = [
        {
            "data": [
            {
            "type": "pie",
            "domain": {
            "x": [
                0,
                1
                ],
            "y": [
                0,
                1
            ]
            },
            "marker": {
                "colors": [
                "#7fc97f",
                "#beaed4",
                "#fdc086"
            ]
            },
            "textinfo": "label+value",
            "labels": genre_names,
            "values": genre_counts, 
            "showlegend": False
            }
            ],
            "layout": {
                "title": "Messages by Genre"
            }
        },
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Number of messages"
                },
                'xaxis': {
                    'title': "Category"
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
    """Load the model and use it to predict the category of the input message"""

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
    """Run the app. Select host and port."""

    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()