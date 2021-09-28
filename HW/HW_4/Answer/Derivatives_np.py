import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# def f(x):
#     return x**2
# 
# r_change = 0.001
# x = 4
# numerical_gradient = (f(x+ r_change)-f(x-r_change))/(2*r_change)
# =============================================================================


def func(x,A,B):
    return A*x**2+ B*x

xlist = np.linspace(0,10,10)
ylist = (func(xlist,2,1.4))

plt.figure(figsize = (10,5))
plt.plot(xlist,ylist,"o", label = "Function")

def d(xlist,ylist):
    yprime = np.diff(ylist) / np.diff(xlist)
    xprime = []
    
    for i in range(len(yprime)):
        xtemp= (xlist[i+1] + xlist[i])/2
        xprime = np.append(xprime,xtemp)
    return xprime,yprime


xprime,yprime = d(xlist,ylist)
plt.plot(xprime,yprime,"o", label = "1s: derivative")
plt.legend()
