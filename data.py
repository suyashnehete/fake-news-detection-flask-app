import numpy as np
import pandas as pd
from matplotlib.pyplot import show, colormaps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

fileName = 'cleaned_data.csv'


def get_data():
    np.random.seed(42)
    df = pd.read_csv(fileName)
    x_train, x_test, y_train, y_test = train_test_split(df.text, df.label, test_size=0.2)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)
    return tfidf_train, tfidf_test, y_train, y_test


def get_vector_transform():
    np.random.seed(42)
    df = pd.read_csv(fileName)
    x_train, x_test, y_train, y_test = train_test_split(df.text, df.label, test_size=0.2)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_vectorizer.fit_transform(x_train)
    return tfidf_vectorizer


def plot(e, x, y):
    plot_confusion_matrix(e, x, y, colorbar=False, cmap="GnBu")
    show()
