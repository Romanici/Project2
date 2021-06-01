
## Project 2 : Disaster Response Pipeline Project
### Udacity nanodegree


This project analyses messages posted publicly  on internet by users during catastrophies and tries to predict if the user really needs help or not. There are several categories to detail which kind of help is needed (water, food, etc.)

---

### Description

The project is made up of three scripts:
- **process_data.py** cleans the data and save it in a SQlite datavase (output_etl.db).
- **train_classifier.py** trains a random forest with multioutput (model.pkl).
- **run.py** loads the Flask app and executes the ML trained model when needed. 

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



