from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
from mma import mmasub
# MAIN DRIVER
def main(nelx,nely,volfrac,penal,rmin,ft):
    print("Inverter mechanism design with MMA")
    print("ndes: " + str(nelx) + " x " + str(nely))
    print("volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal))
    print("Filter method: " + ["Sensitivity based","Density based"][ft])
    # Max and min stiffness
    Emin=1e-9
    Emax=100.0
    kspring=1.0
    dof_in=0
    dof_out=2*(nely+1)*nelx
    solid=np.union1d(
        np.array([x*nely+y for x in range(5) for y in range(3)]),
        np.array([x*nely+y for x in range(nelx-5,nelx) for y in range(3)])
    )
    # dofs:
    ndof = 2*(nelx+1)*(nely+1)
	# Allocate design variables (as array), initialize and allocate sens.
    x=volfrac * np.ones(nely*nelx,dtype=float)
    xold1=x.copy()
    xold2=x.copy()
    xPhys=x.copy()
    low=np.zeros((nely*nelx,1))
    upp=np.ones((nely*nelx,1))
    # FE: Build the index vectors for the for coo matrix format.
    KE=lk()
    edofMat=np.zeros((nelx*nely,8),dtype=int)
    for elx in range(nelx):
	    for ely in range(nely):
		    el = ely+elx*nely
		    n1=(nely+1)*elx+ely
		    n2=(nely+1)*(elx+1)+ely
		    edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
    # Construct the index pointers for the coo format
    iK = np.kron(edofMat,np.ones((8,1))).flatten()
    jK = np.kron(edofMat,np.ones((1,8))).flatten()    
    # Filter: Build (and assemble) the index+data vectors for the coo matrix format
    nfilter=int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc=0
    for i in range(nelx):
	    for j in range(nely):
		    row=i*nely+j
		    kk1=int(np.maximum(i-(np.ceil(rmin)-1),0))
		    kk2=int(np.minimum(i+np.ceil(rmin),nelx))
		    ll1=int(np.maximum(j-(np.ceil(rmin)-1),0))
		    ll2=int(np.minimum(j+np.ceil(rmin),nely))
		    for k in range(kk1,kk2):
			    for l in range(ll1,ll2):
				    col=k*nely+l
				    fac=rmin-np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
				    iH[cc]=row
				    jH[cc]=col
				    sH[cc]=np.maximum(0.0,fac)
				    cc=cc+1
	# Finalize assembly and convert to csc format
    H=coo_matrix((sH,(iH,jH)),shape=(nelx*nely,nelx*nely)).tocsc()	
    Hs=H.sum(1)
	# BC's and support
    dofs=np.arange(2*(nelx+1)*(nely+1))
    fixed=np.union1d(dofs[1:2*nelx*(nely+1)+1:2*(nely+1)],np.array([2*(nely+1)-2,2*(nely+1)-1]))
    free=np.setdiff1d(dofs,fixed)
	# Solution and RHS vectors
    f=np.zeros((ndof,1))
    u=np.zeros((ndof,1))
    adj=np.zeros((ndof,1))
	# Set load
    f[dof_in,0]=1
    dobjdu=np.zeros((ndof,1))
    dobjdu[dof_out,0]=1
	# Initialize plot and plot the initial design
    plt.ion() # Ensure that redrawing is possible
    fig,ax = plt.subplots()
    im = ax.imshow(-xPhys.reshape((nelx,nely)).T, cmap='gray',\
    interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
    fig.show()
   	# Set loop counter and gradient vectors 
    loop=0
    change=1
    dv   = np.ones(nely*nelx)
    dobj = np.ones(nely*nelx)
    while change>0.01 and loop<200:
        loop=loop+1
		# Setup and solve FE problem
        sK=((KE.flatten()[np.newaxis]).T*(Emin+(xPhys)**penal*(Emax-Emin))).flatten(order='F')
        K = coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsc()
        # Add springs at input and output nodes
        K[dof_in,dof_in] += kspring
        K[dof_out,dof_out] += kspring
        # Remove constrained dofs from matrix
        K = K[free,:][:,free]
		# Solve system 
        u[free,0]=spsolve(K,f[free,0])    
		# Objective and sensitivity
        obj = u[dof_out,0]
        adj[free,0]=spsolve(K,-dobjdu[free,0])    
        dobj[:]=(penal*xPhys**(penal-1)*(Emax-Emin))*(np.dot(adj[edofMat].reshape(nelx*nely,8),KE) * u[edofMat].reshape(nelx*nely,8) ).sum(1)
        dv[:] = np.ones(nely*nelx)
		# Sensitivity filtering:
        if ft==0:
            dobj[:] = np.asarray((H*(x*dobj))[np.newaxis].T/Hs)[:,0] / np.maximum(0.001,x)
        elif ft==1:
            dobj[:] = np.asarray(H*(dobj[np.newaxis].T/Hs))[:,0]
            dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:,0]
        # MMA iteration
        m = 1 # number of constraints
        a0mma, amma, cmma, dmma = 1, np.zeros((m,1)), 1000*np.ones((m,1)), np.zeros((m,1))
        (xnew,_,_,_,_,_,_,_,_,low,upp)=mmasub(m,nely*nelx,loop,
                                              x[np.newaxis].T,np.zeros(nely*nelx)[np.newaxis].T,np.ones(nely*nelx)[np.newaxis].T,
                                              xold1[np.newaxis].T,xold2[np.newaxis].T,
                                              obj,dobj[np.newaxis].T,
                                              np.array([[sum(xPhys)/(volfrac*nely*nelx)-1]]),dv[np.newaxis],
                                              low,upp,a0mma,amma,cmma,dmma)
        xold2[:]=xold1
        xold1[:]=x
        x[:]=xnew[:,0]
		# Filter design variables
        if ft==0:   xPhys[:]=x
        elif ft==1:	xPhys[:]=np.asarray(H*x[np.newaxis].T/Hs)[:,0]
        xPhys[solid]=1.0
		# Compute the change by the inf. norm
        change=np.linalg.norm(x.reshape(nelx*nely,1)-xold1.reshape(nelx*nely,1),np.inf)
		# Plot to screen
        xPlot = np.round(xPhys).reshape((nelx,nely))
        xPlot = np.hstack([np.fliplr(xPlot), xPlot])
        im.set_array(-xPlot.T)
        fig.canvas.draw()
		# Write iteration history to screen (req. Python 2.6 or newer)
        print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(\
				    loop,obj,sum(xPhys)/(nelx*nely),change))
	# Make sure the plot stays and that the shell remains	
    plt.show()
    input("Press any key...")
#element stiffness matrix
def lk():
    E=1
    nu=0.3
    k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
    KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
    [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
    [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
    [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ]);
    return (KE)
# The real main driver    
if __name__ == "__main__":
    # Default input parameters
    nelx=180
    nely=60
    volfrac=0.4
    rmin=5.4
    penal=3.0
    ft=1 # ft==0 -> sens, ft==1 -> dens
    import sys
    if len(sys.argv)>1: nelx   =int(sys.argv[1])
    if len(sys.argv)>2: nely   =int(sys.argv[2])
    if len(sys.argv)>3: volfrac=float(sys.argv[3])
    if len(sys.argv)>4: rmin   =float(sys.argv[4])
    if len(sys.argv)>5: penal  =float(sys.argv[5])
    if len(sys.argv)>6: ft     =int(sys.argv[6])
    main(nelx,nely,volfrac,penal,rmin,ft)
