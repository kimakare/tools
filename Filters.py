from numpy import pi
from control import tf

s = tf('s')

def num(tf):
    [[num]] = tf.num
    n = len(num)
    ret = 0
    for i, a in enumerate(num):
        ret = ret + a*s**(n-1-i)
    return ret

def den(tf):
    [[den]] = tf.den
    n = len(den)
    ret = 0
    for i, a in enumerate(den):
        ret = ret + a*s**(n-1-i)
    return ret

def LP(fc):
    wc = 2*pi*fc
    return 1/(s/wc+1)

def HP(fc):
    wc = 2*pi*fc
    return s/(s+wc)

def DIF(fc):
    wc = 2*pi*fc
    return s/(s/wc+1)

def PL(f1,f2):
    w1 = 2*pi*f1
    w2 = 2*pi*f2
    return (s/w1+1)/(s/w2+1)

def PD(f1,f2):
    w1 = 2*pi*f1
    w2 = 2*pi*f2
    return (s+w1)/(s+w2)

def Notch(f0,Q,d):
    num = s**2+2/10**(d/20)/Q*2*3.14*f0*s+(2*3.14*f0)**2
    den = s**2+2/Q*2*3.14*f0*s+(2*3.14*f0)**2
    return num/den

def Peak(f0,Q,d):
    num = s**2+2/Q*2*3.14*f0*s+(2*3.14*f0)**2
    den = s**2+2/Q/10**(d/20)*2*3.14*f0*s+(2*3.14*f0)**2
    return num/den

def BHP(fc):
    wc = 2*pi*fc
    num = s**7 + 7*wc*s**6 + 21*wc**2*s**5 + 35*wc**3*s**4
    den = s**7 + 7*wc*s**6 + 21*wc**2*s**5 + 35*wc**3*s**4 + 35*wc**4*s**3 + 21*wc**5*s**2 + 7*wc**6*s + wc**7
    return num/den

def BLP(fc):
    wc = 2*pi*fc
    num = 35*wc**4*s**3 + 21*wc**5*s**2 + 7*wc**6*s + wc**7
    den = s**7 + 7*wc*s**6 + 21*wc**2*s**5 + 35*wc**3*s**4 + 35*wc**4*s**3 + 21*wc**5*s**2 + 7*wc**6*s + wc**7
    return num/den
