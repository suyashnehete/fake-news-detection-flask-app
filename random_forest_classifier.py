import pickle

from sklearn.ensemble import RandomForestClassifier

from data import get_data, plot

x_train, x_test, y_train, y_test = get_data()

clf = RandomForestClassifier(n_estimators=200)
clf.fit(x_train, y_train)
print("{0:.2f}%".format(clf.score(x_test, y_test) * 100))
plot(clf, x_test, y_test)

pickle.dump(clf, open("random_forest_model.pkl", 'wb'))
