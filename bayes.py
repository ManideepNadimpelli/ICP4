import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

data = pd.read_csv("glass.csv")
X_train, X_test = train_test_split(data, test_size=0.5, random_state=int(time.time()))
gnb = GaussianNB()

used_features = [
    "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"
]


gnb.fit(
    X_train[used_features].values,
    X_train["Type"]
)

y_pred = gnb.predict(X_test[used_features])

print(y_pred.shape[0])

print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
    .format(
    X_test.shape[0],
    (X_test["Type"] != y_pred).sum(),
    100 * (1 - (X_test["Type"] != y_pred).sum() / X_test.shape[0])
))