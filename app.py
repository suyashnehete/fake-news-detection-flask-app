import pickle

import pandas as pd
from flask import Flask, render_template, request

from data import get_vector_transform

app = Flask(__name__)

tfidf_vectorizer = get_vector_transform()

random_forest = pickle.load(open("random_forest_model.pkl", "rb"))
svm = pickle.load(open("svm.pkl", "rb"))
decision_tree = pickle.load(open("decision_tree.pkl", "rb"))
logistic_regression = pickle.load(open("logistic_regression.pkl", "rb"))
naive_bayes = pickle.load(open("naive_bayes.pkl", "rb"))
passive_aggresive = pickle.load(open("passive_aggresive.pkl", "rb"))
sgd = pickle.load(open("sgd.pkl", "rb"))


@app.route('/')
def main():
    return render_template('main.html', hasData=False)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    url = request.get_data(as_text=True)[5:]
    temp_df2 = pd.Series([url], name="text")
    tfidf_test = tfidf_vectorizer.transform(temp_df2)
    lst = [
        random_forest.predict(tfidf_test)[0],
        svm.predict(tfidf_test)[0],
        decision_tree.predict(tfidf_test)[0],
        logistic_regression.predict(tfidf_test)[0],
        naive_bayes.predict(tfidf_test)[0],
        passive_aggresive.predict(tfidf_test)[0],
        sgd.predict(tfidf_test)[0]
    ]
    return render_template(
        'main.html',
        hasData=True,
        result="Fake" if lst.count("REAL") < lst.count("FAKE") else "Real",
        real="{:.2f}%".format(lst.count("REAL")/len(lst)*100),
        fake="{:.2f}%".format(lst.count("FAKE")/len(lst)*100),
        rfc=random_forest.predict(tfidf_test)[0],
        svc=svm.predict(tfidf_test)[0],
        dtc=decision_tree.predict(tfidf_test)[0],
        lrc=logistic_regression.predict(tfidf_test)[0],
        nbc=naive_bayes.predict(tfidf_test)[0],
        pac=passive_aggresive.predict(tfidf_test)[0],
        sgc=sgd.predict(tfidf_test)[0]
    )


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
