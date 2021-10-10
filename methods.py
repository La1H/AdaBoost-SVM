#import library
import numpy as np
from cvxopt import matrix,solvers

# methods
# Constructing matrix's for P, q, G, h, A, B like construct of quadratic programming
'''
Min: 1/2 * x.T.dot(P.dot(x))+q.T.dot(x)
s.t. G*x <= h
    Ax = b
'''

def dual_problem_quadratic_program(X, y, C = None, distribution_weights = None):
    ''' Solve with soft svm '''
    '''    With X shaped (N, d)
        y shaped (N, )
        C is a he so danh gia do rong le, is scale
        distribution_weights is a adaboost weight
    '''                                
    N, d = X.shape
    # Nhan y vao tung phan tu cua X (Nhan 2 ma tran chu khong nhan ma tran voi vecto)
    # Khai bao P, q
    yX = X * y[:, np.newaxis]
    P = yX.dot(yX.T)
    P = matrix(P)
    q = matrix(np.ones((N, 1))* -1)
    # Build A, b, G, h
    ''' Moi phan tu cua Lamda co 0< lam < CW
    suy ra co 2 ma tran don vi G tren chua moi phan tu > 0 
    Ma tran don vi G duoi chua moi phan tu < CW'''

    # Xét truong hop hard-SVM khong co trong so
    if C is None:
        G = matrix(-1 * np.eye(N))
        h = matrix(np.zeros(N))
    # Xet truong hop soft- SVM     
    else:
        G = matrix(np.vstack(((np.eye(N) * -1), np.eye(N))))
        if distribution_weights is None:
            h = matrix(np.hstack((np.zeros(N), np.ones(N) * C)))
        else:
            h = matrix(np.hstack((np.zeros(N), distribution_weights * (np.ones(N) * C))))

    
    y = y.astype('float')
    A = matrix(y[:,np.newaxis].reshape(1,-1))
    b = matrix(np.zeros(1))
    return P, q, G, h, A, b



#solvers.qp(P, q, G, h, A, b)
def dual_problem_quadratic_solver(P, q, G, h, A, b):
    solvers.options['show_progress'] = False
    return solvers.qp(P, q, G, h, A, b)




# Dua ra Lamda, shape (N , 1)
def svm_lagrange_mutipliers(solution):
    return np.array(solution['x'])
# Tim nhung Lamda co gia tri khac 0
def svm_support_vectors(lamda):
    return np.where(lamda >= 1e-2)[0]

# solve weight of svm
def svm_weight(X, y, lamda):
    return np.dot(X.T,(y[:,np.newaxis]*lamda)).flatten() # shape(d, )

# solve bias of svm
def svm_bias(X, y, S, weight):
    return np.mean(y[S] - np.dot(X[S],weight))

# predict svm
def svm_pred(X, w, b):
    return np.sign(X.dot(w)+b)

# accuracy of svm
def svm_accuracy(pred, y):
    '''pred shaped (N, )
    y shaped (N, )'''
    return np.mean((y== pred))



