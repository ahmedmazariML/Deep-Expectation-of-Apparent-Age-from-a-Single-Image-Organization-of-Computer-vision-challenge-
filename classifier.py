from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
import pickle

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


class Classifier(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
	estimator = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight='balanced', verbose=0, random_state=None, max_iter=1000)
	estimator.fit(X, y)
	self.clf =  CalibratedClassifierCV(base_estimator=estimator, method='isotonic', cv=10)
	self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X) # The classes are in the order of the labels returned by get_classes
        
    def get_classes(self):
        return self.clf.classes_
        
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self
