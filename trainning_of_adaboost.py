import numpy as np
import methods
import svm
def fit(X, y, M = 10, C = None , instance_categorization = False):
    '''
    Input:
        X: data
        y: label
        M: Adaboost loops
        instance_categorization is  boolean which means use or not use  instance categorization
    Output H is a models of adaboosts , which is sign func of sum of M loops SVM
    '''
    #Xac dinh number of data va length of feature
    N, d = X.shape
    #initial weight adjustment and instance categorization
    #W_ada = methods.intinitialization_instance_categorization(N)
    W_ada = methods.intinitialization_weight_adjustment(X, y)
    #W_ada = methods.intinitialization_weight_adjustment(N)
    #Creat list of each models svm after adaboost
    w = []
    b = []
    #creat list of cofident
    alpha = []
    if instance_categorization is True:
        C_ada = methods.intinitialization_instance_categorization(N)
        for i in range(M):
            # Creat model
            WC = W_ada * C_ada
            wi, bi = svm.fit(X, y, C , distribution_weight= WC)
            # Append wi and bi to the list
            w.append(wi)
            b.append(bi)
            #predict the model
            pred_i = methods.predict_svm(X, wi, bi)
            # Find true, false index after training svm
            true_index, false_index = methods.find_true_false_index(y, pred_i)
            # Compute i-th confident and append to the alpha
            alpha_i = methods.confident(W_ada,false_index)
            alpha.append(alpha_i)
            # Update weight adjustment and instance categorization
            W_ada = methods.update_weight_adjustment(W_ada, alpha_i,true_index, false_index)
            C_ada = methods.update_instance_categorization(X, y, wi, bi)
    else:
        for i in range(M):
            # Creat model
            wi, bi = svm.fit(X, y, C,distribution_weight = W_ada)
            # Append wi and bi to the list 
            w.append(wi)
            b.append(bi)
            # Predict the model 
            pred_i = methods.predict_svm(X, wi, bi)
            # Find true, false index after training svm
            true_index, false_index = methods.find_true_false_index(y, pred_i)
            # Compute i_th confident and append to the alpha
            alpha_i = methods.confident(W_ada,false_index)
            alpha.append(alpha_i)
            # Update weight adjustment
            W_ada = methods.update_weight_adjustment(W_ada, alpha_i,true_index,false_index)
    return w, b, alpha    
            

def predict(X,  w, b, alpha,M =10 ):
    H = np.zeros(X.shape[0])
    for i in range (M):
        H += alpha[i]*(X.dot(w[i]) +b[i])
    return np.sign(H)