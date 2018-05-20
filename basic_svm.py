import numpy as np
import math
from sklearn import preprocessing,model_selection as cross_validation, neighbors, svm
import pandas as pd

df = pd.read_csv('ba.csv')

X = np.array(df.drop(['y'], 1))
y = np.array(df['y'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.02, random_state=6)

clf = svm.SVC()

clf.fit(X_train, y_train)

confidence = clf.score(X_test, y_test)

print('%3f'%confidence)