import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

if __name__ == "__main__":
    # This statement imports the bank dataset
    df = pd.read_csv('bank-full.csv', sep=';', header=None)

    # The next statement display the scope of the dataset
    print(f'The dataset is {df.shape[0]} rows long and has {df.shape[1] - 1}'
          f' attributes along with a target variable.')

    # These two statements split the dataset into the target column
    # And a new dataset without the target column
    target = df.iloc[:, -1:]
    df1 = df.iloc[:, :-1]

    # y: the target column
    # x: the dataset minus the target column
    y = target
    x = df1

    # Splitting the data - 80:20 ratio
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print("Training split input- ", X_train.shape)
    print("Testing split input- ", X_test.shape)

    # A simple decision tree classifier
    dTree = DecisionTreeClassifier()

    # This chunk of code normalizes the data to be numerical instead of categorical
    enc = preprocessing.OrdinalEncoder()
    enc.fit(X_train)
    X_train1 = enc.transform(X_train)

    # This chunk of code normalizes the data to be numerical instead of categorical
    enc.fit(X_test)
    X_test1 = enc.transform(X_test)

    dTree.fit(X_train1, y_train)
    print('Decision Tree Classifier Created')

    y_pred = dTree.predict(X_test1)
    print("Classification report - \n", classification_report(y_test, y_pred))

