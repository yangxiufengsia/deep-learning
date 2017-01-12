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
#print X_new.shape
#print X_new[0,2]
print np.matrix(y).shape
print X_new.shape

'''## gradient decent for choosing paramter w'''
w0_ini=1.0
w1_ini=1.0
w2_ini=0.7
w_ini=np.matrix([w0_ini,w1_ini,w2_ini])
print w_ini.shape
j=1.0
j_ini=1.0
#K=(1/(1+np.exp(-(X_new*w_ini.T)))-np.matrix(y).T).T
#j_new=0.0
#print K.shape
#print np.matrix(X_new[:,0]).shape
#s=(1/200.0)*(1/(1+np.exp(-(X_new*w_ini.T)))-np.matrix(y).T).T*X_new[:,0]
#print s
l=np.matrix([0.05/200.0,0.05/200.0,0])
I=np.eye(3,3)
reg=I*l.T
print reg.shape
#print reg.shape
i=1
while abs(j)>0.0:
    #w_new=w_ini-0.2*(1/200)*(1/1+np.exp(-(X_new*w_ini))-y).T*X_new[:,0]
    #w0_new=w0_ini-0.05*(1/200.0)*(1/(1+np.exp(-X_new*w_ini.T))-np.matrix(y).T).T*np.matrix(X_new[:,0]).T
    #w1_new=w1_ini-0.05*(1/200.0)*(1/(1+np.exp(-X_new*w_ini.T))-np.matrix(y).T).T*np.matrix(X_new[:,1]).T
    #w2_new=w2_ini-0.05*(1/200.0)*(1/(1+np.exp(-X_new*w_ini.T))-np.matrix(y).T).T*np.matrix(X_new[:,2]).T
    w_new=w_ini-(0.05*(1/200.0)*(1/(1+np.exp(-X_new*w_ini.T))-np.matrix(y).T).T*np.matrix(X_new)+w_ini*reg)
    #print w_new
    #print w0_new.tolist(),w1_new.tolist(),w2_new.tolist()
    #print (w0_new,w1_new,w2_new)
    #w_new=np.matrix([w0_new.tolist(),w1_new.tolist(),w2_new.tolist()])
    #w0_ini=w0_new
    #w1_ini=w1_new
    #w2_ini=w2_new
    w_ini=w_new
    j_new=(-0.05/200.0)*(np.matrix(y)*np.log(1/(1+np.exp(-(X_new*w_ini.T))))+(1-np.matrix(y))*np.log(1-1/(1+np.exp(-(X_new*w_ini.T)))))
    +(1/400.0)*(w_ini[0,0]**2+w_ini[0,1]**2)
    j=abs(j_ini)-abs(j_new)
    j_ini=j_new

    #print j_new,j_ini
    i=i+1
    print "loss after" +str(i) + "iteration:"+str(j_ini)


ye=1/(1+np.exp(-(X_new[11,:]*w_ini.T)))
#print np.matrix(y)[0,11]
#print ye
xtest=np.linspace(-2,2)
ytest=-xtest*w_ini[0,0]/w_ini[0,1]+w_ini[0,2]/w_ini[0,1]
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.plot(xtest,ytest)
plt.show()
#boundary=X_new*w_ini.T
#plt.plot(boundary)
#plt.show()
