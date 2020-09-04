# Disaster Response Pipeline Project

https://github.com/ciaran-grant/Udacity-Disaster-Response

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to https://view6914b2f4-3001.udacity-student-workspaces.com/


### Project Summary:
This is a submission for the Disaster Response Pipeline Project for the Udacity Data Scientist Nanodegree: https://www.udacity.com/course/data-scientist-nanodegree--nd025

There are two piplines which will produce a web app.

The ETL pipeline follows Extract, Transform, Load (/data/process_data.py).
- Extract relevant .csv files provided (data/disaster_messages.csv, data/disaster_categories.csv)
- Tranform data by separating categories, merging and removing duplicate rows
- Load clean data to SQL database (data/DisasterResponse.db)

The ML pipeline tokenizes the text data and trains a multiclass random forest classificator (models/train_classifier.py)
- Tokenization includes case normalisation, punctuation removal, tokenisation, stop word removal, stemming and lemmatisation.
- Random Forest classifier is trained using default hyperparameters (Grid Search is available)
- Model is saved with pickle (models/classifier.pkl)

Access the web app at the above address.
