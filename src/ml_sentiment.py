import string

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure required NLTK corpora are present in any deployment (e.g., DigitalOcean)
NLTK_RESOURCES = {
    "stopwords": "corpora/stopwords",
    "punkt": "tokenizers/punkt",
    "wordnet": "corpora/wordnet",
    "vader_lexicon": "sentiment/vader_lexicon",
    "punkt_tab": "tokenizers/punkt_tab",
}


def ensure_nltk_resources():
    for resource, path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


ensure_nltk_resources()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    token = word_tokenize(text)
    filtered = []
    for i in token:
        if i not in stop_words:
            filtered.append(lemmatizer.lemmatize(i))
    
    return ' '.join(filtered)
    
def read_file(filename):
    df = pd.read_csv(filename)
    df.columns = ['Sentiment','Text','Score']
    return df 

def emotion_score(X_train, y_train, X_test):
    vector = CountVectorizer()
    X_train2 = vector.fit_transform(X_train)
    X_test2 = vector.transform(X_test)

    model = LinearRegression()
    model.fit(X_train2,y_train)

    score = model.predict(X_test2)
    return score

def prebuilt_model(X_test):
    analyzer = SentimentIntensityAnalyzer()
    predictions = []
    for i in X_test:
        score = analyzer.polarity_scores(i)
        score1 = score['compound']
        if score1 >= 0.2:
            predictions.append('positive')
        elif score1 <= -0.2:
            predictions.append('negative')
        else:
            predictions.append('neutral')
    return np.array(predictions)

def evaluate_model(X,y,model,type=0,k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    precisions = []
    recalls = []
    confusions = []
    f1_scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if type == 0:
            pred = my_model(X_train, y_train, X_test, model)
        elif type == 1:
            pred = prebuilt_model(X_test)

        accuracy = accuracy_score(y_test,pred)
        precision = precision_score(y_test,pred, average=None)
        recall = recall_score(y_test,pred,average=None)
        confusion = confusion_matrix(y_test,pred)
        f1 = f1_score(y_test, pred, average=None)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        confusions.append(confusion)
        f1_scores.append(f1)

    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions, axis=0)
    avg_recall = np.mean(recalls, axis=0)
    sum_confusion = np.sum(confusions, axis=0)
    avg_f1 = np.mean(f1_scores, axis=0) 
    return avg_accuracy, avg_precision, avg_recall, sum_confusion, avg_f1

def my_model(X_train, y_train, X_test, model_name):
    vector = CountVectorizer()
    X_train2 = vector.fit_transform(X_train)
    X_test2 = vector.transform(X_test)

    name = str(model_name).strip().lower()
    if name in ('naive bayes', 'naivebayes', 'nb', 'multinomialnb'):
        clf = MultinomialNB()
    elif name in ('svm', 'support vector machine', 'support-vector-machine'):
        clf = SVC()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    clf.fit(X_train2, y_train)
    pred = clf.predict(X_test2)
    return pred


