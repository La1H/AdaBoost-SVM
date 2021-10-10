import numpy as np
from data.Vertebral_column import load_data
# import trainning_of_adaboost as toa
import svm
from sklearn.svm import SVC 
from sklearn.metrics import classification_report
import trainning_of_adaboost as toa
#test SVM

X_train, X_test,y_train, y_test = load_data()
w, b = svm.fit(X_train, y_train, C = 100)
test_pred = np.sign(X_test.dot(w)+b)
print(w, b)



