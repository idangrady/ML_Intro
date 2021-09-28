import numpy as np



# =============================================================================
# add regularization
# j = \gamma * ||w||^2 + 1/n * np.sum(for i in range(n) {0,1 - y_i * (2*x_i -b)})
# =============================================================================


# =============================================================================
# Gradient:
#     if y_i*f(x) >=1:
#         dj_i/ dw_k = 2\gamma*w_k 
#         dj_i/db = 0
#     else:
#         dj_i/dw_k = 2\gammaw_k - y_i*x_i
#         dj_i/db= y_i
# =============================================================================


# =============================================================================
# update role: 
#     w w-a*dw
#     b = b-a*db
# =============================================================================


class SVM:
    def __init__(self, learning_rate = 0.001, lamda_param = 0.01, n_iters=1000):
        self.lr = learning_rate
        self.lamda_para =lamda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
        
    def fit(self,X,y):
        d,n =X.shape
        y_ = np.where(y<=0 ,-1,1)
        
        self.w = np.zeros((d,1))
        self.b = np.zeros((1,1))
        
        for _ in range(self.n_itersi):
            for idx, x_i in enumerate(X):
                condition =  (y_[idx] * ((np.dot(self.w, x_i)- self.b) ) >=1)
                if condition:
                    self.w -= 2*(self.lr * self.w) * self.lamda_param
                    
                else:
                    self.w -= self.lr(2*self.lamda_para * self.w - y_[idx] * ((np.dot(x_i,y_[idx])) -self.b)) 
                    self.b -=self.lr (y_[idx])

            
    
    def predict(self,x):
        linear_output = np.dot(self.w, x) - self.b
        return np.sign(linear_output)