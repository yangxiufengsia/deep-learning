import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
# Generate a dataset and plot it
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
#print y.shape
#print X.shape
#h=1/1+np.exp(-w.T*X)
#plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
#plt.show()

## add one column with 1 value to X data
ad=np.ones((200,1))
#print ad.shape
X_new=np.append(X,ad,1)

"""set activation function for input data"""
a1=np.matrix(X_new)
"""initialization"""
w1=np.matrix(np.random.randn(3,5))
w2=np.matrix(np.random.randn(6,1))
#b1=np.matrix(np.ones((1,5)))
#b2=np.matrix(np.ones((1,1)))
#w2o=np.tile(w2,200)
#print w2o.shape

j_ini=3.0#(-1/200.0)*(np.matrix(y)*np.log(a3)+(1-np.matrix(y))*np.log(1-a3))
print j_ini
j=1.0
i=1
for i in range(0,10000):

    """calculate the forword Variables"""
    z2=a1*w1
    #print z2.shape
    a2=1/(1+np.exp(-z2))
    a2_new=np.append(a2,ad,1)
    z3=a2_new*w2
    a3=1/(1+np.exp(-z3))
    '''calculate the error in last layer'''
    delta3=a3.T-y
    '''backpropagation calculation'''
    delta2=np.multiply(w2*delta3,np.multiply(a2_new,(1-a2_new)).T)
    #print delta2.shape
    #print delta3.shape

    """gradient of the cost function calculation"""
    dw2=a2_new.T*delta3.T
    dw1=a1.T*delta2.T
    #db2=delta3
    #db1=delta2
    #print dw1
    #print dw2.shape
    #print dw1.shape
    test=np.delete(dw1,5,1)
    #print test
    """ optimize the hyper-parameters """
    j_new=(-1/200.0)*(np.matrix(y)*np.log(a3)+(1-np.matrix(y))*np.log(1-a3))
    j=abs(j_ini)-abs(j_new)
    j_ini=j_new


    w_new2=w2-0.01*dw2
    w_new1=w1-0.01*test
    w2=w_new2
    w1=w_new1
    #b2=b2-0.05*db2
    #print b2.shape
    #b1=b1-0.05*db1

    #print w2
    #print w1
    i=i+1
    print "loss after" +str(i) + "iteration:"+str(j_ini)

""" predict """
w1tr=w1
w2tr=w2
z2tr=a1*w1tr
a2tr=1/(1+np.exp(-z2tr))
a2ntr=np.append(a2tr,ad,1)
z3tr=a2ntr*w2tr
a3tr=1/(1+np.exp(-z3tr))

print a3tr
print y
#print
# Set min and max values and give it some padding
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = 0.01
# Generate a grid of points with distance h between them
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
print xx,yy
#print xx.shape,yy.shape
# Predict the function value for the whole gid
#Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
#print Z.shape
#Z = Z.reshape(xx.shape)
#print Z.shape
# Plot the contour and training examples
#plt.contourf(xx, yy, a3tr, cmap=plt.cm.Spectral)
#plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
#plt.show()

#print X_new.shape
#print X_new[0,2]
