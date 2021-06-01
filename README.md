
## Project 2 : Disaster Response Pipeline Project
### Udacity nanodegree


This project analyses messages posted publicly on internet during catastrophies and tries to predict if the users really needs help or not. There are several categories that are used to detail which kind of help is needed (water, food, etc.). 
Using this app, organizations can detect easily the people who need help during an emergency, saving time and resources. 

---

### Description

The three main scripts of the project are:
- **process_data.py** cleans the data and save it in a SQlite datavase (output_etl.db). Using the command line:
```python3 process_data.py "disaster_messages.csv" "disaster_categories.csv" "output_etl.db"``` 
- **train_classifier.py** trains a random forest with multioutput (model.pkl). Using the command line:
```python3 train_classifier.py "../data/output_etl.db" "model.pkl"```
- **run.py** loads the Flask app and executes the ML trained model when needed. Using the command line: ```python3 "run.py"```

To start up the web app successfully you need to:
1. Run **process_data.py** using as arguments the filepaths of the messages, categories and database files. 
2. Run **train_classifier.py.py** using as arguments the filepaths of the database you want to load and the ML model you want to save. 
3. Execute **run.py** and go to http://localhost:3001/

File structure:
```
WORKSPACE
| app
| | - template
| | |- master.html # main page of web app
| | |- go.html     # classification result page of web app
| |- run.py        # Flask file that runs app
| data
| |- disaster_categories.csv # data to process
| |- disaster_messages.csv   # data to process
| |- process_data.py
| |- output_etl.db.          # database to save clean data to
| models
| |- train_classifier.py
| |- model.pkl             # saved model
|
| ETL Pipeline Preparation.ipynb
| ML Pipeline Preparation.ipynb
| README.md
```

### Dependencies

- Python (3.8)
- Machine Learning libraries
  - NumPy
  - Pandas
  - Scikit-Learn
  - Joblib 
- Natural Language Processing libraries
  - NLTK
- Database libraries
  - SQLAlchemy
- Python Web App and visualization
  - Flask
  - Plotly



