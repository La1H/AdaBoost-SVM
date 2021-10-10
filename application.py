import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
import pandas as pd
import svm

def fit(X, y, C = None, distribution_weight = None):
    # preprocess: 
    N, d = X.shape

    # Obtain Quadratic Programming
    P, q, G, h, A, b = svm.dual_problem_quadratic_program(X, y, C, distribution_weight)

    #Solve Quadratic Program
    sol = svm.dual_problem_quadratic_solver(P, q,G, h, A, b)

    # Caculate Lagrange 
    lam = svm.svm_lagrange_mutipliers(sol)

    # Find Svm suport vectors that lam > 0
    S = svm.svm_support_vectors(lam)

    # Find weight
    w = svm.svm_weight(X, y, lam)

    # Find bias 
    b = svm.svm_bias(X, y, S, w)
    return w ,b 

