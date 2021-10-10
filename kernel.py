import numpy as np


class kernel:
    def __init__(self, kernel_name , gamma):
        self.kernel_name = kernel_name
        self.gamma = gamma
    def kernel(self, xi, xj):
        try:
            if self.kernel_name == 'rbf':
                return self.rbf(xi, xj)
        except:
            print('kerner is not exitst')
    def rbf(self, xi, xj):
        return np.exp(-self.gamma * (np.linalg.norm(xi - xj) ** 2))

