import pickle

from sklearn.naive_bayes import MultinomialNB

from data import get_data, plot

x_train, x_test, y_train, y_test = get_data()

clf = MultinomialNB()
clf.fit(x_train, y_train)
print("{0:.2f}%".format(clf.score(x_test, y_test) * 100))
plot(clf, x_test, y_test)

pickle.dump(clf, open("naive_bayes.pkl", 'wb'))
