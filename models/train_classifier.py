import sys
import pandas as pd
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import pickle

def load_data(database_filepath):
    # Access sql database
    engine = create_engine('sqlite:///' + database_filepath, pool_pre_ping=True)
    
    # Read table from connection
    df = pd.read_sql_table(table_name = 'database', con = engine)

    # Define message as feature column
    X = df['message']
    # Define remaining categories as response
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
        
    # case normalisation
    text = text.lower()
    
    # punctuation removal
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenisation
    tokens = word_tokenize(text)
    
    # stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    # stemming and lemmatisation
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]
    tokens = [PorterStemmer().stem(w) for w in tokens]
    
    return tokens


def build_model():
    # Build pipeline with tokenizer, TF-IDF and MultiClassOutput with RandomForestClassifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Grid Search takes a while to run/test, so removed for time being.
    #parameters = {
    #    'clf__estimator__max_depth' : [2, 3]  
    #    'clf__estimator__min_samples_leaf' : [100, 200]
    #}

    #model = GridSearchCV(pipeline, param_grid=parameters)
    
    model = pipeline # only takes 2 mins without grid search..
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    # Get test predictions
    Y_pred = model.predict(X_test)
    
    # Build classification reports for each category
    for category in range(len(category_names)):
        print(Y_test.columns[category])
        print(classification_report(Y_test.iloc[:, category], Y_pred[:, category]))
    pass


def save_model(model, model_filepath):
    # Save classifier model with pickle
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


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