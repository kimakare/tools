import numpy as np
from numpy import pi
from matplotlib import pyplot as plt
from matplotlib import patches as pat

def svd(mat):
    # S = Uh@mat@V
    if mat.ndim == 2:
        U, _, Vh = np.linalg.svd(mat)
        Uh = U.conj().T
        V = Vh.conj().T
    elif mat.ndim == 3:
        U = np.zeros(mat.shape,np.complex128)
        Vh = np.zeros(mat.shape,np.complex128)
        for i in range(mat.shape[0]):
            U[i], _, Vh[i] = np.linalg.svd(mat[i])
        Uh = U.conj().transpose(0,2,1)
        V = Vh.conj().transpose(0,2,1)
    else: raise Exception('Dimention of matrix is not valid')
    return Uh, V

def matdet(mat):
    if mat.ndim == 3:
        if len(mat[0]) != len(mat[0][0]): raise Exception('Error: Cannot define inverse, invalid matrix shape')
        ret = np.zeros(len(mat),np.complex128)
        for i in range(len(mat)):
            ret[i] = np.linalg.det(mat[i])
    elif mat.ndim == 2:
        if len(mat) != len(mat[0]): raise Exception('Error: Cannot define inverse, invalid matrix shape')
        ret = np.linalg.det(mat)
    else: raise Exception('Error: dimention is not valid')
    return ret

def mineig(data):
    if data.ndim == 2:
        eig = np.linalg.eigvals(data)
        ret = eig[np.argmin(np.abs(eig))]
    elif data.ndim == 3:
        ret = np.zeros(data.shape[0],np.complex128)
        for i in range(ret.shape[0]):
            eig = np.linalg.eigvals(data[i])
            ret[i] = eig[np.argmin(np.abs(eig))]
    return ret

def maxeig(data):
    if data.ndim == 2:
        eig = np.linalg.eigvals(data)
        ret = eig[np.argmax(np.abs(eig))]
    elif data.ndim == 3:
        ret = np.zeros(data.shape[0],np.complex128)
        for i in range(ret.shape[0]):
            eig = np.linalg.eigvals(data[i])
            ret[i] = eig[np.argmax(np.abs(eig))]
    return ret

def plotMaxEigLoci(data,xbound=[None,None],ybound=[None,None]):
    eig = maxeig(data)
    plt.figure(figsize=(8,6))
    plt.scatter(np.real(eig),np.imag(eig),color="black")
    plt.scatter([-1],[0],color="mediumblue")
    plt.xlim(*xbound)
    plt.ylim(*ybound)
    plt.xlabel("real part",fontsize=15)
    plt.ylabel("imaginary part",fontsize=15)
    plt.title("nyquist plot",fontsize=20)
    plt.show()

def plotEigLoci(data,xbound=[None,None],ybound=[None,None]):
    eig = np.zeros((data.shape[0],data.shape[1]),np.complex128)
    for i in range(data.shape[0]):
        eig[i] = np.linalg.eigvals(data[i])
    plt.figure(figsize=(8,6))
    for i in range(data.shape[1]):
        plt.scatter(np.real(eig[:,i]),np.imag(eig[:,i]),color='black')
    plt.scatter([-1],[0],color="mediumblue")
    plt.xlim(*xbound)
    plt.ylim(*ybound)
    plt.xlabel("real part",fontsize=15)
    plt.ylabel("imaginary part",fontsize=15)
    plt.title("nyquist plot",fontsize=20)
    plt.show()

def gershgorin_disks(data):
    ret = np.zeros((data.shape[0],data.shape[1],2),np.complex128)
    for i in range(len(data)):
        ret[i,:,0] = np.diag(data[i])
        ret[i,:,1] = np.sum(np.abs(data[i]-np.diag(ret[i,:,0])),axis=1)
    return ret

def plotGershgorinDisks(data):
    disks = gershgorin_disks(data)
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes()
    for disk in disks:
        for i in range(disk.shape[0]):
            c = pat.Circle(xy=(np.real(disk[i][0]),np.imag(disk[i][0])),radius=disk[i][1],edgecolor="black",fill=False,linewidth=0.5)
            ax.add_patch(c)
    ax.scatter([-1],[0],color="mediumblue")
    ax.set_xlabel("real part",fontsize=15)
    ax.set_ylabel("imaginary part",fontsize=15)
    ax.set_title("gershgorin disks",fontsize=20)
    fig.show()

def generalized_gershgorin_disks(data):
    ret = np.zeros((data.shape[0],data.shape[1],2),np.complex128)
    for i in range(data.shape[0]):
        M = np.zeros(data[0].shape)
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                if j == k: M[j][k] = 0.
                else: M[j][k] = np.abs(data[i][j][k]/data[i][k][k])
        l = maxeig(M)
        ret[i,:,0] = np.diag(data[i])
        ret[i,:,1] = l*np.abs(ret[i,:,0])
    return ret

def plotGeneralizedGershgorinDisks(data):
    disks = generalized_gershgorin_disks(data)
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes()
    for disk in disks:
        for i in range(disk.shape[0]):
            c = pat.Circle(xy=(np.real(disk[i][0]),np.imag(disk[i][0])),radius=disk[i][1],edgecolor="black",fill=False,linewidth=0.5)
            ax.add_patch(c)
    ax.scatter([-1],[0],color="mediumblue")
    ax.set_xlabel("real part",fontsize=15)
    ax.set_ylabel("imaginary part",fontsize=15)
    ax.set_title("gershgorin disks",fontsize=20)
    fig.show()

def RGA(data):
    if data.ndim==2:
        ret = np.zeros(data.shape,np.complex128)
        ret = data*((np.linalg.inv(data)).T)
    elif data.ndim==3:
        ret = np.zeros(data.shape,np.complex128)
        for i in range(len(data)):
            ret[i] = data[i]*((np.linalg.inv(data[i])).T)
    else: raise Exception('Argument has not eough dimention')
    return ret

def RGAnum(data):
    rga = RGA(data)
    if rga.ndim==2:
        ret = 0
        tmp = rga - np.diag([ 1 for _ in range(len(rga))])
        for i in range(len(rga)):
            for j in range(len(rga[0])):
                ret = ret + np.abs(tmp[i][j])
    elif rga.ndim==3:
        ret = np.zeros(len(rga))
        for i in range(len(rga)):
            tmp = rga[i] - np.diag([ 1 for _ in range(len(rga[0]))])
            for j in range(len(rga[0])):
                for k in range(len(rga[0][0])):
                    ret[i] = ret[i] + np.abs(tmp[j][k])
    else: raise Exception('Argument does not have eough dimention')
    return ret

def plotRGA(freq,data):
    ynum = len(data[0])
    xnum = len(data[0][0])
    fig, axis = plt.subplots(xnum,ynum,figsize=(18,15),constrained_layout=True,sharex=True,sharey=True)
    rga = RGA(data)
    shapedrga = rga.transpose(1,2,0)
    for y in range(ynum):
        for x in range(xnum):
            axis[y][x].loglog(freq,np.abs(shapedrga[y][x]), color="crimson", zorder=2)
            axis[y][x].grid(which="major", linewidth=2)
            axis[y][x].grid(which="minor", linewidth=1, linestyle=":")
            axis[y][x].tick_params(size=20,direction='in')
    axis[0][0].legend()
    fig.suptitle('RGA plot',fontsize=30)
    fig.supylabel('RGA (absolute)',fontsize=30)
    fig.supxlabel('frequency [Hz]',fontsize=30)
    fig.show()

def plotRGAnum(freq,*data):
    plt.figure(figsize=(8,6))
    for i in range(len(data)):
        if len(data) == 1: plt.semilogx(freq,RGAnum(data[i]), color="crimson", zorder=2)
        else: plt.semilogx(freq,RGAnum(data[i]), zorder=2, label=(str(i+1)+"th"))
    plt.legend()
    plt.grid(which="major", linewidth=1)
    plt.grid(which="minor", linewidth=0.5, linestyle=":")
    plt.xlabel("frequency (Hz)",fontsize=15)
    plt.ylabel("RGA number",fontsize=15)
    plt.title("RGA number",fontsize=20)

def plotCoherence(freq,*data):
    plt.figure(figsize=(12,4))
    for i in range(len(data)):
        if len(data) == 1: plt.semilogx(freq,data[i], color="crimson", zorder=2)
        else: plt.semilogx(freq,data[i], zorder=2, label=(str(i+1)+"th"))
    plt.legend()
    plt.grid(which="major", linewidth=1)
    plt.grid(which="minor", linewidth=0.5, linestyle=":")
    plt.xlabel("frequency (Hz)",fontsize=15)
    plt.ylabel("Coherence",fontsize=15)
    plt.title("Coherence",fontsize=20)

def plotMIMObode(freq,*tf):
    tf = np.array(tf)
    ynum = len(tf[0][0])
    xnum = len(tf[0][0][0])
    shapedtf = tf.transpose(0,2,3,1)
    fig, axis = plt.subplots(xnum,ynum,figsize=(24,12),constrained_layout=True,sharex=True,sharey=True)
    for y in range(ynum):
        for x in range(xnum):
            for i in range(len(tf)):
                if len(tf) == 1: axis[y][x].semilogx(freq,20*np.log10(np.abs(shapedtf[i][y][x])),color='crimson')
                else: axis[y][x].semilogx(freq,20*np.log10(np.abs(shapedtf[i][y][x])),label=(str(i+1)+"th"))
            axis[y][x].grid(which="major", linewidth=2)
            axis[y][x].grid(which="minor", linewidth=1, linestyle=":")
            axis[y][x].tick_params(size=20,direction='in')
    axis[0][0].legend()
    fig.suptitle('bode plot',fontsize=30)
    fig.supylabel('magnitude [dB]',fontsize=30)
    fig.supxlabel('frequency [Hz]',fontsize=30)
    fig.show()

def plotSingularValue(freq,tf):
    _, s, _ = np.linalg.svd(tf)
    plt.figure(figsize=(12,4),tight_layout=True)
    for i in range(len(tf[0])):
        plt.semilogx(freq,20*np.log10(np.abs(s[:,i])),zorder=3,linewidth=3.0,label=(str(i+1)+"th"))
    plt.ylabel("magnitude [dB]",fontsize=15)
    plt.grid(which="major", linewidth=2)
    plt.grid(which="minor", linewidth=1, linestyle=":")
    plt.legend(fontsize=10)
    plt.xlabel("frequency [Hz]",fontsize=15)
    plt.title("singular value plot",fontsize=20)
    plt.show()

def plotEigenValue(freq,tf):
    _, s = np.linalg.eig(tf)
    plt.figure(figsize=(12,4),tight_layout=True)
    for i in range(len(tf[0])):
        plt.semilogx(freq,20*np.log10(np.abs(s[:,i])),zorder=3,linewidth=3.0,label=(str(i+1)+"th"))
    plt.ylabel("magnitude [dB]",fontsize=15)
    plt.grid(which="major", linewidth=2)
    plt.grid(which="minor", linewidth=1, linestyle=":")
    plt.legend(fontsize=10)
    plt.xlabel("frequency [Hz]",fontsize=15)
    plt.title("singular value plot",fontsize=20)
    plt.show()

def plotSISObode(freq,*tf):
    fig, ax = plt.subplots(2,1,figsize=(12,6),tight_layout=True)
    for i in range(len(tf)):
        if len(tf)==1: ax[0].semilogx(freq,20*np.log10(np.abs(tf[i])),color="crimson",zorder=3,linewidth=3.0)
        else: ax[0].semilogx(freq,20*np.log10(np.abs(tf[i])),zorder=3,linewidth=3.0,label=(str(i+1)+"th"))
    ax[0].set_ylabel("magnitude [dB]",fontsize=15)
    ax[0].grid(which="major", linewidth=2)
    ax[0].grid(which="minor", linewidth=1, linestyle=":")
    ax[0].legend(fontsize=10)
    for i in range(len(tf)):
        if len(tf)==1: ax[1].semilogx(freq,np.angle(tf[i],deg=True),zorder=3,linewidth=3.0,color="crimson")
        else: ax[1].semilogx(freq,np.angle(tf[i],deg=True),zorder=3,linewidth=3.0)
    ax[1].set_ylabel("phase [degree]",fontsize=15)
    ax[1].grid(which="major", linewidth=2)
    ax[1].grid(which="minor", linewidth=1, linestyle=":")
    fig.supxlabel("frequency [Hz]",fontsize=15)
    fig.suptitle("bode plot",fontsize=20)
    fig.show()

def plotSISONyquist(tf,xbound=[-10,10],ybound=[-10,10]):
    plt.figure(figsize=(8,6))
    plt.plot(np.real(tf),np.imag(tf),color="crimson")
    plt.scatter([-1],[0],color="mediumblue")
    plt.xlim(*xbound)
    plt.ylim(*ybound)
    plt.xlabel("real part",fontsize=15)
    plt.ylabel("imaginary part",fontsize=15)
    plt.title("nyquist plot",fontsize=20)
    plt.show()

def plotMIMONyquist(tf,xbound=[-10,10],ybound=[-10,10]):
    det = matdet(tf)
    plt.figure(figsize=(8,6))
    plt.plot(np.real(det),np.imag(det),color="crimson")
    plt.scatter([0],[0],color="mediumblue")
    plt.xlim(*xbound)
    plt.ylim(*ybound)
    plt.xlabel("real part",fontsize=15)
    plt.ylabel("imaginary part",fontsize=15)
    plt.title("nyquist plot",fontsize=20)
    plt.show()
