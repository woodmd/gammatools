import numpy as np
from scipy.optimize import curve_fit

def cov_rho(cov,i,j):
    """Return the i,j correlation coefficient of the input covariance
    matrix."""
    return cov[i,j]/np.sqrt(cov[i,i]*cov[j,j])

def cov_angle(cov,i,j):
    """Return error ellipse rotation angle for parameters i,j."""
    return 0.5*np.arctan((2*cov_rho(cov,i,j)*np.sqrt(cov[i,i]*cov[j,j]))/
                         cov[j,j]-cov[i,i])
    

def fit_svd(x,y,yerr,n=2):
    """
    Linear least squares fitting using SVD.  Finds the set of
    parameter values that minimize chi-squared by solving the matrix
    equation:
    
    a = (A'*A)^(-1)*A'*y
    
    where A is the m x n design matrix, y is the vector of n data
    points, and yerr is a vector of errors/weights for each data
    point.  When the least squares solution is not unique the SVD
    method finds the solution with minimum norm in the fit parameters.
    The solution vector is given by
    
    a = V*W^(-1)*U'*y
    
    where U*W*V' = A is the SVD decomposition of the design matrix.
    The reciprocal of singular and/or small values in W are set to
    0 in this procedure.
    """

    x = np.array(x)
    y = np.array(y)
    yerr = np.array(yerr)
    
    A = np.zeros(shape=(len(x),n+1))

    for i in range(n+1):
        A[:,i] = np.power(x,i)/yerr

    b = y/yerr
    (u,s,v) = np.linalg.svd(A,full_matrices=False)
    ub = np.sum(u.T*b,axis=1)
    a = np.sum(ub*v.T/s,axis=1)

    vn = v.T/s
    cov = np.dot(vn,vn.T)

    return a, cov

    cov2 = np.zeros((2,2))

    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            for k in range(2):
                cov2[i,j] += v[i,k]*v[j,k]/s[k]**2

    print 'cov'
    print cov

    print 'cov2'
    print cov2

    return a, cov2


np.random.seed(1)

def fn(x,a,b):
    return a + b*x

#x = np.array([-10.0,-5.0,10.0])
x = np.array([1.0,2.0,3.0,4.0])
y = fn(x,1.0,-1.0) + 0.2*np.random.normal(size=len(x))
yerr = 0.2*np.ones(len(x))



def chi2_fn(f,x,y,yerr):
    return np.sum((f(x)-y)**2/yerr**2,axis=2)

#print np.polyfit(x,y,1,w=yerr,cov=True)

print 'Fit 1'
a, cov = curve_fit(fn,x,y,sigma=1./yerr)
print a
print cov
print rho(cov,0,1)

print 'Fit 2'
a, cov = curve_fit(fn,x,y,sigma=yerr)
print a
print cov
print rho(cov,0,1)

print 'Fit SVD'
a, cov = fit_svd(x,y,yerr,2)
#cov[0,1] *= -1
#cov[1,0] *= -1

npoint = 1000

ax, ay = np.meshgrid(np.linspace(-3.0,3,npoint),np.linspace(-3.0,3.0,npoint),
                     indexing='ij')

import matplotlib.pyplot as plt

chi2 = chi2_fn(lambda t: fn(t,ax.reshape(npoint,npoint,1),
                            ay.reshape(npoint,npoint,1)),
               x.reshape(1,1,len(x)),
               y.reshape(1,1,len(x)),
               yerr.reshape(1,1,len(x)))

chi2 = chi2-chi2_fn(lambda t: fn(t,a[0].reshape(1,1,1),
                                 a[1].reshape(1,1,1)),
                    x.reshape(1,1,len(x)),
                    y.reshape(1,1,len(x)),
                    yerr.reshape(1,1,len(x)))

print np.min(chi2)


plt.figure()

plt.errorbar(x,y,yerr=yerr)
plt.plot(x,fn(x,*a))

plt.figure()

plt.contour(ax,ay,chi2,levels=[0,1.0,2.3])

plt.errorbar(a[0],a[1],
             xerr=np.sqrt(cov[0,0]),
             yerr=np.sqrt(cov[1,1]),
             marker='o')

plt.gca().grid(True)


rho01 = rho(cov,0,1)


print 'rho ', rho01

angle = np.pi - 0.5*np.arctan((2*rho01*np.sqrt(cov[0,0]*cov[1,1]))/
                              cov[1,1]-cov[0,0])

print angle

t = np.linspace(-1,1,100)

#plt.plot(a[0]+t*np.cos(angle),a[1]+t*np.sin(angle))


print a
print np.sqrt(cov[0,0]), np.sqrt(cov[1,1])
print cov

plt.show()

#print chi2(lambda t: f(t,*a+),
