import numpy as np
import copy

class Vector3D(object):

    
    
    def __init__(self,x):
        self._x = x

    def separation(self,v):      
        costh = np.sum((self._x.T*v._x.T).T,axis=0)        
        return np.arccos(costh)
    
    def norm(self):
        return np.sqrt(np.sum(self._x*self._x,axis=0))

    def theta(self):
        return np.arctan2(np.sqrt(self._x[0]*self._x[0] +
                                  self._x[1]*self._x[1]),self._x[2])

    def phi(self):
        return np.arctan2(self._x[1],self._x[0])    

    def lat(self):
        return np.pi/2.-self.theta()

    def lon(self):
        return self.phi()
    
    def normalize(self):
        self._x *= 1./self.norm()

    def rotatey(self,angle):
        yaxis = Vector3D(angle*np.array([0.,1.,0.]))
        self.rotate(yaxis)

    def rotatez(self,angle):
        zaxis = Vector3D(angle*np.array([0.,0.,1.]))
        self.rotate(zaxis)
        
    def rotate(self,axis):

        angle = axis.norm()

        if np.abs(angle) == 0: return
        
        tmp = np.zeros(self._x.shape)
        tmp = np.transpose(tmp.T[:] + axis._x)

        eaxis = Vector3D(tmp)
        eaxis._x *= 1./angle
        par = np.sum(self._x*eaxis._x,axis=0)
        
        paxis = Vector3D(copy.copy(self._x))

        paxis._x -= par*eaxis._x

        cp = eaxis.cross(paxis)
        
        self._x = par*eaxis._x + np.cos(angle)*paxis._x + np.sin(angle)*cp._x

        
    def cross(self,axis):
        x = np.zeros(self._x.shape)
        
        x[0] = self._x[1]*axis._x[2] - self._x[2]*axis._x[1]
        x[1] = self._x[2]*axis._x[0] - self._x[0]*axis._x[2]
        x[2] = self._x[0]*axis._x[1] - self._x[1]*axis._x[0]
        return Vector3D(x)

    def project(self,v):

        yaxis = Vector3D(np.array([0.,1.,0.]))
        zaxis = Vector3D(np.array([0.,0.,1.]))

        yaxis *= -v.theta()
        zaxis *= -v.phi()
        
        vp = Vector3D(copy.copy(self._x))

        vp.rotate(zaxis)
        vp.rotate(yaxis)
        
        return vp

    
    @staticmethod
    def createLatLon(lat,lon):

        return Vector3D.createThetaPhi(np.pi/2.-lat,lon)
        
    @staticmethod
    def createThetaPhi(theta,phi):

        x = np.array([np.sin(theta)*np.cos(phi),
                      np.sin(theta)*np.sin(phi),
                      np.cos(theta)*(1+0.*phi)])

        return Vector3D(x)
        
#        v = []        
#        for i in range(len(x)):
#            v.append(Vector3D(x[i]))
#        return v

    def __getitem__(self, i):
        return Vector3D(self._x[:,i])

    def __mul__(self,x):

        self._x *= x
        return self
    
    def __str__(self):
        return self._x.__str__()

if __name__ == '__main__':

    lat = np.array([1.0])#,0.0,-1.0,0.0])
    lon = np.array([0.0])#,1.0, 0.0,-1.0])
    
    v0 = Vector3D.createLatLon(0.0,np.radians(2.0))
    v1 = Vector3D.createLatLon(np.radians(lat),np.radians(lon))

    print 'v0: ', v0
    print 'v1: ', v1
    
    v2 = v1.project(v0)

    y = -np.degrees(v2.theta()*np.cos(v2.phi()))
    x = np.degrees(v2.theta()*np.sin(v2.phi()))

    for i in range(4):    
        print '%.3f %.3f'%(x[i], y[i])
