
import numpy as np

def row_average_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (m,1) array where each entry is the average of a row
    """
    x_1 = np.array(np.mean(x[:,0:1])).reshape((1,1))
    d,n = x.shape
    for i in range(1,n):
        x_i = np.array(np.mean(x[:,i:i+1])).reshape((1,1))
        x_1= np.concatenate((x_1,x_i),axis = 1)
    return x_1.T

def cv(value_list):
    return np.transpose(rv(value_list))

# Takes a list of numbers and returns a row vector: 1 x n
def rv(value_list):
    return np.array([value_list])


def top_bottom_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (2,1) array where the first entry is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    # top part how do we compute the top? we compute on the row
    row,col = x.shape
    upper_part = row//2
    under = row-upper_part
    a_0 = np.array(np.mean(x[:1,:])).reshape((1,1))
    for i in range(1,upper_part):
        a_i = np.array(np.mean(x[i:i+1,:])).reshape((1,1))
        a_0 = np.concatenate((a_0,a_i),axis = 1)
    a_0 = np.mean(a_0)
    a_under = np.array(np.mean(x[upper_part:upper_part+1,:])).reshape((1,1))
    for l in range(upper_part+1,row):
        x_un = np.array((np.mean(x[l:l+1,:])).reshape((1,1)))
        a_under = np.concatenate((a_under,x_un))
    a_under = np.mean(a_under)
    return np.array((a_0,a_under)).reshape((2,1))


ans=top_bottom_features(np.array([[1,2,3],[3,9,2],[2,1,9]])).tolist()



ans=top_bottom_features(np.array([[1,2,3],[3,9,2]])).tolist()

x = np.array([(1,2,3),(4,5,6),(7,8,9)])
print(row_average_features(x))