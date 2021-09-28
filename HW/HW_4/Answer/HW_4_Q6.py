import numpy as np

def package_ans(gd_vals):
    x, fs, xs = gd_vals
    return [x.tolist(), [fs[0], fs[-1]], [xs[0].tolist(), xs[-1].tolist()]]


def cv(value_list):
    return np.transpose(rv(value_list))

def rv(value_list):
    return np.array([value_list])

def f1(x):
    return float((2 * x + 3)**2)

def df1(x):
    return 2 * 2 * (2 * x + 3)

def f2(v):
    x = float(v[0]); y = float(v[1])
    return (x - 2.) * (x - 3.) * (x + 3.) * (x + 1.) + (x + y -1)**2

def df2(v):
    x = float(v[0]); y = float(v[1])
    return cv([(-3. + x) * (-2. + x) * (1. + x) + \
               (-3. + x) * (-2. + x) * (3. + x) + \
               (-3. + x) * (1. + x) * (3. + x) + \
               (-2. + x) * (1. + x) * (3. + x) + \
               2 * (-1. + x + y),
               2 * (-1. + x + y)])
        
def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y


def gd(f,df, x0, step_size_fn, max_iter):
    xs=[]
    fs = []
    x_i = x0
    d,n  = x0.shape
    xs.append(x_i)
    for i in range(max_iter):
        if(d<=1):
            fs_result = f1((x_i))
            de_f  = df1(x_i)            
        else:
            fs_result = f2((x_i))
            de_f  = df2(x_i)
            
        step_s = step_size_fn(0)
        new_x = x_i - (step_s* de_f)
        fs.append(fs_result)
        x_i = new_x  
        xs.append(new_x)
        
    return (new_x,fs,xs)


        
def num_grad(f, delta=0.001):
    def df(x):
        d,n =x.shape
        if(d==1):
            df = np.array((f(x+delta)- f(x-delta))/(2*delta)).reshape((1,1))
        else:
            dd_1= []
            for i in range(d):  
                zer= np.zeros((n+1,1))
                zer[i] = delta
                df = np.array((f(x+zer)- f(x-zer))/(2*delta))
                dd_1.append(df)
            df = np.array(dd_1)
        return df
    return df

def gd2(f, df, x0, step_size_fn, max_iter):
    prev_x = x0
    fs = []; xs = []
    for i in range(max_iter):
        prev_f= f((prev_x))
        prev_grad = df(prev_x)
        fs.append(prev_f); xs.append(prev_x)
        if i == max_iter-1:
            return prev_x, fs, xs
        step = step_size_fn(i)
        prev_x = prev_x - step * prev_grad

def hinge(v):
    return np.where(v >= 1, 0, 1 - v)

# x is dxn, y is 1xn, th is dx1, th0 is 1x1
def hinge_loss(x, y, th, th0):
    return hinge(y * (np.dot(th.T, x) + th0))

# =============================================================================
#     v = (y*(np.dot(th.T,x)+ th0))
#     return(hinge(v))
# =============================================================================

# x is dxn, y is 1xn, th is dx1, th0 is 1x1, lam is a scalar
def svm_obj(X, y, th, th0, lam):
    return np.mean(hinge_loss(X, y, th, th0)) + lam * np.linalg.norm(th) ** 2# =============================================================================
#     d,n = x.shape
#     score = 0
#     for i in range(n):
#         x_i = x[:,i:i+1]
#         y_i = y[:,i:i+1]
#         score+= hinge_loss(x_i,y_i,th,th0)
#     score = score/n 
#     norm_th = lam*((th**2)**2)
#     return ((score+norm_th))
# =============================================================================


# Returns the gradient of hinge(v) with respect to v.
def d_hinge(v):
    return np.where(v >= 1, 0, -1)

# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th
def d_hinge_loss_th(x, y, th, th0):
    step1 = d_hinge(y * (np.dot(th.T, x) + th0))
    output = step1 * y *x
    return  output

# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th0
def d_hinge_loss_th0(x, y, th, th0):
    output = d_hinge(y*(np.dot(th.T, x) + th0))
    return  output* y

# Returns the gradient of svm_obj(x, y, th, th0) with respect to th
def d_svm_obj_th(x, y, th, th0, lam):
    return np.mean(d_hinge_loss_th(x, y, th, th0)) + lam * (th * 2)

# Returns the gradient of svm_obj(x, y, th, th0) with respect to th0
def d_svm_obj_th0(x, y, th, th0, lam):
    return np.mean(d_hinge_loss_th0(x, y, th, th0))
           
# Returns the full gradient as a single vector (which includes both th, th0)
def svm_obj_grad(X, y, th, th0, lam):
    grad_th = d_svm_obj_th(X, y, th, th0, lam)
    grad_th0 = d_svm_obj_th0(X, y, th, th0, lam)
    return np.vstack([grad_th, grad_th0])




def minimize(f, x0, step_size_fn, max_iter):
    return gd2(f,num_grad(f), x0, step_size_fn, max_iter)

if __name__ == "__main__":
        
    # Test case 1
    #ans=package_ans(gd(f1, df1, cv([0.]), lambda i: 0.1, 1000))
    
    # Test case 2
    #ans=package_ans(gd(f2, df2, cv([0., 0.]), lambda i: 0.01, 1000))
    
# =============================================================================
#     #numerical Derivatives:
# 
# #    ans=(num_grad(f1)(x).tolist(), x.tolist())
#     x = cv([0.1, -0.1])
#     ans=(num_grad(f2)(x).tolist(), x.tolist())
# 
#     print(ans)
# =============================================================================
    
# =============================================================================
#     #minimizing
#     ans = package_ans(minimize(f2, cv([0., 0.]), lambda i: 0.01, 1000))
#     print(ans)
# =============================================================================

# =============================================================================
#     #implementation
#     sep_e_separator = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])
#     # Test case 1
#     x_1, y_1 = super_simple_separable()
#     th1, th1_0 = sep_e_separator
#     ans=svm_obj(x_1, y_1, 0.1*th1, th1_0, 0.0)
#     print(ans)
# =============================================================================
    X1 = np.array([[1, 2, 3, 9, 10]])
    y1 = np.array([[1, 1, 1, -1, -1]])
    th1, th10 = np.array([[-0.31202807]]), np.array([[1.834     ]])
    X2 = np.array([[2, 3, 9, 12],
                   [5, 2, 6, 5]])
    y2 = np.array([[1, -1, 1, -1]])
    th2, th20=np.array([[ -3.,  15.]]).T, np.array([[ 2.]])
    print((th2))
# =============================================================================
#     d_hinge(np.array([[ 71.]])).tolist()
#     d_hinge(np.array([[ -23.]])).tolist()
#     d_hinge(np.array([[ 71, -23.]])).tolist()
# =============================================================================
# =============================================================================
#     
#    # d_hinge_loss_th(X2[:,0:1], y2[:,0:1], th2, th20).tolist()
#     print(d_hinge_loss_th(X2, y2, th2, th20).tolist())
# 
#     d_hinge_loss_th0(X2, y2, th2, th20).tolist()
#     
#     
#     print(d_svm_obj_th(X2[:,0:1], y2[:,0:1], th2, th20, 0.01).tolist())
#     d_svm_obj_th(X2, y2, th2, th20, 0.01).tolist()
#    # d_svm_obj_th0(X2[:,0:1], y2[:,0:1], th2, th20, 0.01).tolist()
#     d_svm_obj_th0(X2, y2, th2, th20, 0.01).tolist()
#     
# =============================================================================
    print(d_hinge_loss_th0(X2, y2, th2, th20).tolist())    
    d_hinge_loss_th0(X2[:,0:1], y2[:,0:1], th2, th20).tolist()
    
    svm_obj_grad(X2, y2, th2, th20, 0.01).tolist()
    svm_obj_grad(X2[:,0:1], y2[:,0:1], th2, th20, 0.01).tolist()