from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_classifier(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)