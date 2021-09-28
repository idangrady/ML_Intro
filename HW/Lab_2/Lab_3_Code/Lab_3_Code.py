import numpy as np


np.set_printoptions(suppress=True)


def perceptron(data, labels, params = {}, hook = None):
    # if T not in params, default to 100
    if(type(data) is list): 
        data = np.array(data)
    if(type(labels) is list):
        labels = np.array(labels)
    T = params.get('T', 50)
    (d, n) = data.shape
    th = np.zeros((d,1)); th_0 = np.zeros((1,1))
    mistake =0

    
    for i in range(T):
        for j in range(n):
            x = data[:,j:j+1]
            y = labels[:,j:j+1]
            
            if(y*(np.dot(th.T,x)+ th_0)>0):
                pass
            else:
                mistake+=1
                th += (-y)*x
                th_0 += y
                if hook: hook((th, th_0))
    
    return (th, th_0,mistake)

def perceptron_through_origin(data,th,target, labels, params = {}, hook = None):
    # if T not in params, default to 100
   
    T = params.get('T', 50)
    (d, n) = data.shape
    #th = np.zeros((d,1)); th_0 = np.zeros((1,1))
    
    mistake =0
    while True:
        for j in range(n):
            x = data[:,j:j+1]
            y = labels[:,j:j+1]
            
            if(y*(np.dot(th.T,x))>0):
                pass
            else:
                mistake+=1
                th += y*x
                #if hook: hook((th))
                if(th== target):break
    
    return (th,mistake)
def calculate_the_min_margin(x,th,th_0,y):
    d,n =x.shape
    min_val =10000000000000
    for i in range(n):
        x_i = x[:,i:i+1]
        current_result = (y[:,i:i+1] * (np.dot(th.T,x_i) + th_0))/np.sqrt(np.sum((th)*th) + th_0**2)
        if(current_result<min_val): min_val =current_result
    return min_val
    
    
def calculated_mistakes(data,labels,th,th_0):
    thetha = np.array([(0),(1),(-0.5)])
     
def upper_bound(data,th,y):
    return((r(data,(0,0))/margin_calculation(data,th,y))**2)

def sign(th, data):
    return(np.sign(np.dot(th,data)))

def margin_calculation(data, th,y):
    d,n = data.shape
    min_margin  =10000000000000
    
    for i in range(n):
        x_i = data[:,i:i+1]
        y_i = labels[:,i:i+1]
        current_d = np.sqrt(np.sum(th**2))
        current_margin = (y_i* (np.dot(th, x_i)))/current_d
        if current_margin<min_margin and current_margin!=0:
            min_margin =current_margin
    return (min_margin)
    
    
def one_off_concat(x,x_i):
    return(np.concatenate((x,x_i), axis = 1))


def one_hot(x,y,k):
    arr = np.zeros((k,1))
    arr[x[0]-1]=y[0]
    for i in range(1,len(x)):
        a =np.zeros((k,1))
        num = x[i]-1
        a[num] = y[i]
        arr = one_off_concat(arr,a)
    return(arr)


def r(data, center):
    pass
    d,n= data.shape
    max_dis =0
    for i in range(n):
        x_i = data[:,i:i+1]
        current_dis = x_i**2
        if current_dis >max_dis: 
            max_dis =current_dis
    return max_dis

def distance_m(th,point):
    x,y = point
    pass

def one_off_scale(data,labels):
    n = data[-1]+1
    ones =np.zeros((n,1))
    for i in range(n-2):
        label_i = labels[0][i]
        loc= data[i]
        ones[loc-1] = label_i
    return ones

def positive(x, th, th0):
    return np.sign(th.T@x + th0)

def perceptron_2(data, labels, params = {}, hook = None):
    # if T not in params, default to 100
    T = params.get('T', 50)
    (d, n) = data.shape
    mistake = 0
    y = labels
    theta = np.zeros((n, 1)); theta_0 = np.zeros((1, 1))
    for t in range(T):
        for i in range(n):
            x = data[:,i:i+1]
            y_i = y.T[i]
            result= (np.dot(x.T, theta)+ theta_0) 
             
            if y_i * (np.dot( theta.T,x,)+ theta_0) <= 0.0:
                theta +=   y_i * x
                theta_0 +=  y_i
                mistake+=1
                if hook: hook((theta, theta_0))
    return theta, theta_0, mistake


if __name__ == "__main__":
    #Samsung = 1, Xiaomi = 2, Sony = 3, Apple = 4, LG = 5, Nokia = 6
    data =  [1, 2, 3, 4, 5, 6]
    labels = [1, 1, -1, -1, 1, 1]
    
# =============================================================================
#     
#     ones= one_off_scale(data,labels)
#     scaler = np.array([[0.001],[1]])
#     new_data = np.concatenate((np.array(data)*scaler,ones),axis = 0)
# =============================================================================
    
   # labels = np.array(labels).reshape((6,1))
    #data_array = np.array(data)one_off_concat
    #data_2 = np.array([[0],[1],[1],[1],[1],0]).reshape((6,1))
    one_off = one_hot(data,labels, 6)
    print(one_off)
    labels = np.array(labels).reshape((1,6))
    #labels = np.array([[-1,1,1,1,1,-1,-1]])
    print(perceptron_2(one_off,labels))
