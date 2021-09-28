import numpy as np



    

class red_blue:
    data = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],
                      [1, 1, 2, 2,  2,  2,  2, 2]])
    labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])
    blue_th = np.array([[0, 1]]).T
    blue_th0 = -1.5 
    red_th = np.array([[1, 0]]).T
    red_th0 = -2.5


class three_point_seperator():
    data = np.array([[1.1, 1, 4],[3.1, 1, 2]])
    labels = np.array([[1, -1, -1]])
    th = np.array([[1, 1]]).T
    th0 = -4
    margin = np.sqrt(2)/2
        
class seperator1():
    data  =np.array([[1,3,4],[1,2,4]])
    lables = np.array([[-1,1,-1]])
    th = np.array([[-0.0737901,2.40847205]]).T
    th0 = -3.492621154916483

class seperator2():
    data  =np.array([[1,3,4],[1,2,4]])
    lables = np.array([[-1,1,-1]])
    th = np.array([[-0.23069578,2.5573501]]).T
    th0 = -3.3857770692522666
    
    
    
class seperator3():
    data  =np.array([[1,2,3,1.5,1,2,3],[1,1,1,1.5,3,3,3]])
    lables = np.array([[1,1,1,-1,-1,-1,-1]])
    th = np.array([[-0.01280916,-142043497]]).T
    th0 = 2.387955038624247
def sum_sc(data, labels,th,th0):
    sum_ = 0
    d,n = data.shape
    for i in range(n):
        x_i = data[:,i:i+1]
        y_i = labels[:,i:i+1]
        sum_ += calcuate_Score(th,th0,x_i,y_i)
    return sum_

def calcuate_Score(th,th0,x_i,y_i):
    return(y_i*(np.dot(th.T,x_i)+th0))

def min_max(data, labels,th,th0,min_max):
    min_ = 10000000000000
    max_ = 0
    d,n = data.shape
    for i in range(n):
        x_i = data[:,i:i+1]
        y_i = labels[:,i:i+1]
        current_score = calcuate_Score(th,th0,x_i,y_i)
        if(min_max =='min'):  
            if current_score < min_:
                min_ =current_score
        elif(min_max =='max'):
            	if current_score > max_:
                    max_ =current_score
    if min_max=='min':
        return min_
    elif min_max =='max':
        return max_


def loss_func(x,y,th,th0,margin):
    score =0
    loss_ =[]
    d,n = x.shape
    for i in range(n):
        x_i = x[:,i:i+1]
        y_i = y[:,i:i+1]
        sqrt = int(np.sum(th**2))
        result = (y_i*(np.dot(th.T,x_i)+th0))/np.sqrt(sqrt)
        if(result<margin):
            current_loss =1-(result)/margin
            score+= (1-result)/margin
            loss_.append(current_loss)
        else:
            loss_.append(0)
        
    print(loss_)


def svm(x,y,th,th0,gamma):
    score =0
    loss_ =[]
    d,n = x.shape
    margin = 2/np.sqrt(int(np.sum(th**2)))
    for i in range(n):
        x_i = x[:,i:i+1]
        y_i = y[:,i:i+1]
        sqrt = np.sqrt(int(np.sum(th**2)))
        result = (y_i*(np.dot(th.T,x_i)+th0))
        if(result<margin):
            current_loss =1-(result)/margin
            score+= ((1-result)) + gamma*(sqrt**2)
    print(score/n)
    
    
#question 2
# =============================================================================
# red_blue = Red_blue()
# blue_th = red_blue.  
# blue_th0 =red_blue.blue_th0
# data = red_blue.data
# labels = red_blue.labels
# 
# max_  = min_max(data,labels,blue_th,blue_th0,'max')
# min_ = min_max(data,labels,blue_th,blue_th0,'min')
# sum_ = sum_sc(data,labels,blue_th,blue_th0)
# print(sum_,min_, max_)
# =============================================================================

# =============================================================================
# # question_3B
# data = three_point_seperator()
# # labels
# loss_func(data.data,data.labels,data.th,data.th0,(np.sqrt(2)/2))
# =============================================================================


#4b
sp_1 = seperator1()
svm(sp_1.data,sp_1.lables,sp_1.th,sp_1.th0,0)