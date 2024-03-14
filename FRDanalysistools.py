import numpy as np
from numpy import pi, sin, cos
from warnings import warn
from control import freqresp

def tf2frd(tf,freq):
    mag, phase, _ = freqresp(tf,2*pi*freq)
    return FRD(mag*(cos(phase)+1j*sin(phase)),freq)

class FRD:
    def __init__(self,data=None,freq=None):
        self.data = data
        self.freq = freq

    def __invert__(self):
        if self.data.ndim == 1:
            res = self.__class__(1/self.data,self.freq)
        elif self.data.ndim == 2:
            res = self.__class__(np.linalg.inv(self.data),self.freq)
        elif self.data.ndim == 3:
            res = self.__class__(np.array([ np.linalg.inv(dataAtEachFreq) for dataAtEachFreq in self.data]),self.freq)
        return res
    
    def __pos__(self):
        return self.__class__(self.data,self.freq)

    def __neg__(self):
        return self.__class__(-self.data,self.freq)

    def __add__(self,other):
        if self.data.ndim == other.data.ndim:
            res = self.__class__(self.data + other.data)
        elif self.data.ndim == 2 and other.data.ndim == 3:
            res = self.__class__(np.array([ self.data + dataAtEachFreq for dataAtEachFreq in other.data]))
        elif self.data.ndim == 3 and other.data.ndim == 2:
            res = self.__class__(np.array([ dataAtEachFreq + other.data for dataAtEachFreq in self.data]))
        else: raise Exception("Addition is defined between 2- or 3- dimention matrices")
        if self.freq is other.freq: res.freq = self.freq
        else: warn("Frequencies does not match")
        return res
    
    def __radd__(self,other):
        if self.data.ndim == 1:
            return self.__class__(np.array([ other + dataAtEachFreq  for dataAtEachFreq in self.data]),self.freq)
        elif self.data.ndim == 2:
            return self.__class__(np.diag([ other for _ in range(self.data.shape[0])]) + self.data,self.freq)
        elif self.data.ndim == 3:
            return self.__class__(np.array([ np.diag([ other for _ in range(self.data.shape[1])]) + dataAtEachFreq for dataAtEachFreq in self.data]),self.freq)
        else: raise Exception("Addition is defined under 3- dimention matrices")
        
    def __sub__(self,other):
        if self.data.ndim == other.data.ndim:
            res = self.__class__(self.data - other.data)
        elif self.data.ndim == 2 and other.data.ndim == 3:
            res = self.__class__(np.array([ self.data - dataAtEachFreq for dataAtEachFreq in other.data]))
        elif self.data.ndim == 3 and other.data.ndim == 2:
            res = self.__class__(np.array([ dataAtEachFreq - other.data for dataAtEachFreq in self.data]))
        else: raise Exception("Subtraction is defined between 2- or 3- dimention matrices")
        if self.freq is other.freq: res.freq = self.freq
        else: warn("Frequencies does not match")
        return res
    
    def __rsub__(self,other):
        if self.data.ndim == 1:
            return self.__class__(np.array([ other - dataAtEachFreq  for dataAtEachFreq in self.data]),self.freq)
        elif self.data.ndim == 2:
            return self.__class__(np.diag([ other for _ in range(self.data.shape[0])]) - self.data,self.freq)
        elif self.data.ndim == 3:
            return self.__class__(np.array([ np.diag([ other for _ in range(self.data.shape[1])]) - dataAtEachFreq for dataAtEachFreq in self.data]),self.freq)
        else: raise Exception("Addition is defined under 3- dimention matrices")

    def __mul__(self,other):
        if self.data.ndim == 1 and other.data.ndim == 1:
            res = self.__class__(self.data * other.data)
        elif self.data.ndim == 2 and other.data.ndim == 2:
            res = self.__class__(self.data @ other.data)
        elif self.data.ndim == 2 and other.data.ndim == 3:
            res = self.__class__(np.array([ self.data @ dataAtEachFreq for dataAtEachFreq in other.data]))
        elif self.data.ndim == 3 and other.data.ndim == 2:
            res = self.__class__(np.array([ dataAtEachFreq @ other.data for dataAtEachFreq in self.data]))
        elif self.data.ndim == 3 and other.data.ndim == 3:
            res = self.__class__(np.array([ self.data[i] @ other.data[i] for i in range(len(self.data))]))
        else: raise Exception("Invalid shape in multiplication")
        if self.freq is other.freq: res.freq = self.freq
        else: warn("Frequencies does not match")
        return res
    
    def __rmul__(self,other):
        return self.__class__(other*self.data,self.freq)

    def __truediv__(self,other):
        res = self.__class__((self * ~other).data)
        if self.freq is other.freq: res.freq = self.freq
        else: warn("Frequencies does not match")
        return res
    
    def __rtruediv__(self,other):
        return other*(~self)
    
    def create_one(dim,freq=None):
        return FRD(np.diag([ 1+0j for _ in range(dim)]),freq)
    
    def create_zero(dim,freq=None):
        return FRD(np.zeros((dim,dim),np.complex128),freq)
    
    def creat_diag(*frd):
        if not all([ frd[0].data.shape == dataAtEachIndex.data.shape for dataAtEachIndex in frd]): raise Exception("Data shape in diagonal line must be same")
        res = FRD(np.array([np.diag([ frdAtEachIndex.data[i] for frdAtEachIndex in frd]) for i in range(frd[0].data.shape[0])]))
        if all([ frd[0].freq is frdAtEachIndex.freq for frdAtEachIndex in frd]): res.freq = frd[0].freq
        else: warn("Frequencies does not match")
        return res
    
    def get_diag(self):
        if self.data.ndim == 2:
            if self.data.shape[0] != self.data.shape[1]: raise Exception("Data is not square")
            res = [self.__class__(self.data[i][i],self.freq) for i in range(self.data.shape[0])]
        if self.data.ndim == 3:
            if self.data.shape[1] != self.data.shape[2]: raise Exception("Data is not square")
            res = [self.__class__(self.data[:,i,i],self.freq) for i in range(self.data.shape[1])]
        return res
    
    def get_data(self):
        return self.data
    
    def get_freq(self):
        return self.freq
