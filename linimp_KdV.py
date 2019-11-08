'''
Implementation of four different schemes applied to the Korteweg-de Vries
equation, as it is given in Eidnes and Li: "Linearly implicit local and
global energy-preserving methods for Hamiltonian PDEs" [EL19].
The schemes are:
** the LILEP and LIGEP schemes of [EL19],
** the LEP and GEP schemes of
Gong, Cai and Wang: "Some new structure-preserving algorithms for general
multi-symplectic formulations of Hamiltonian PDEs" (2014)

Code written by Sølve Eidnes and Lu Li
'''

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

newparams = {'figure.figsize': (8.0, 5.0), 'axes.grid': True,
             'lines.markersize': 8, 'lines.linewidth': 2,
             'font.size': 12}
plt.rcParams.update(newparams)

def newton_KdV_LEP(u,un,eta,gamma,deltax,Ax,k,M,tol):
    f = lambda un: 1/k*Ax**3*(un-u) + 1/6*deltax*Ax*eta*((Ax*un)**2+(Ax*un)*(Ax*u)+(Ax*u)**2) + .5*gamma**2*(deltax**3*un+deltax**3*u)
    J = lambda un: 1/k*Ax**3 + 1/6*deltax*Ax*eta*(spdiags(Ax*u,0,M,M)*Ax+2*spdiags(Ax*un,0,M,M)*Ax) + .5*gamma**2*deltax**3
    err = la.norm(f(un))
    c = 0
    while err > tol:
        un = un - spsolve(J(un),f(un))
        err = la.norm(f(un))
        c = c+1
        if c > 5:
            break
            print('err =',err)
    return un

def newton_KdV_GEP(u,un,eta,gamma,Dx,I,k,M,tol):
    f = lambda un: 1/k*(un-u) + 1/6*np.dot(Dx,un**2+un*u+u**2) + .5*gamma**2*np.dot(np.matmul(Dx,np.matmul(Dx,Dx)),un+u)
    J = lambda un: 1/k*I + 1/6*eta*Dx*(spdiags(u,0,M,M)+2*spdiags(un,0,M,M)) + .5*gamma**2*np.matmul(Dx,np.matmul(Dx,Dx))
    err = la.norm(f(un))
    c = 0
    while err > tol:
        un = un - la.solve(J(un),f(un))
        err = la.norm(f(un))
        c = c+1
        if c > 5:
            break
            print('err =',err)
    return un

def fourierD1(M,L):
    mu = 2*np.pi/L
    D1 = np.zeros([M,M])
    for j in range(M):
        for k in range(M):
            if j == k:
                D1[j,k] = 0
            else:
                D1[j,k] = (1/2*mu*(-1)**(j+k))/np.tan(1/2*mu*(j-k)*L/M)
    if np.mod(M,2) == 1:
        for j in range(int((M-1)/2)):
            for k in range(int(j+(M+1)/2),M):
                D1[j,k] = -D1[j,k]
                D1[M-j-1,M-k-1] = -D1[M-j-1,M-k-1]
    return D1

if __name__ == "__main__":
    k = .01 # Time step size
    T = 1 # End time
    N = int(T/k) # Number of time steps
    
    # Initial conditions:
    #
    # Test problem 1 in [EL19]:
    P = 2 # Length of spatial domain
    M = 200 # Number of spatial steps
    h = P/M # Spatial step size
    x = np.linspace(0,P-h,M)
    gamma = 0.022
    eta = 1
    u0 = np.cos(np.pi*x) # Test problem 1 in [EL19]
    # Test problem 2 in [EL19]:
    '''P = 20 # Length of spatial domain
    M = 100 # Number of spatial steps
    h = P/M # Spatial step size
    gamma = 1
    eta = 6
    c = 4
    u_ana = lambda x,t: .5*c/np.cosh(np.mod(-x+c*t,L)-L/2)**2
    u0 = u_ana(x,0)
    '''
    
    # Creating the average and forward difference matrices:
    e = np.ones(M)
    Ax = .5*spdiags([e,e,e], np.array([-M+1,0,1]), M, M)
    deltax = 1/h*spdiags([e,-e,e], np.array([-M+1,0,1]), M, M)
    I = spdiags(e, 0, M, M)
    
    uNs = np.zeros([4,M]) # To store solution values from all six methods
    # To store discrete approximations to the energy at every time step:
    energies = np.zeros(N)
    energiesK = np.zeros(N)
    
    # LILEP:
    u = u0.copy()
    tic = time.time()
    for i in range(N):
        A = 1/k*Ax**3 + .5*eta*deltax*Ax*spdiags(Ax*u,0,M,M)*Ax + .5*gamma**2*deltax**3
        b = 1/k*Ax**3*u-.5*gamma**2*deltax**3*u
        un = spsolve(A,b)
        energies[i] = h*1/6*sum(-gamma**2*3*(deltax*u)**2 + eta*(Ax*u)**3)
        energiesK[i] = h*1/6*sum(-gamma**2*((deltax*u)**2 + 2*(deltax*u)*(deltax*un)) + eta*(Ax*u)**2*(Ax*un))
        u = un
    print('The LILEP scheme used {:f} seconds.'.format(time.time()-tic))
    uNs[0,:] = u
        
    newparams = {'figure.figsize': (8.0, 5.0), 'axes.grid': True,
                 'lines.markersize': 8, 'lines.linewidth': 1.5,
                 'font.size': 10}
    plt.rcParams.update(newparams)
    plt.plot(x,u,color=(0,.4,1))
    plt.xlabel('$x$')
    plt.ylabel('u(x,T)')
    
    # LEP:
    u = u0.copy()
    tic = time.time()
    for i in range(N):
        un = newton_KdV_LEP(u,u,eta,gamma,deltax,Ax,k,M,1e-10)
        energies[i] = h*1/6*sum(-gamma**2*3*(deltax*u)**2 + eta*(Ax*u)**3)
        energiesK[i] = h*1/6*sum(-gamma**2*((deltax*u)**2 + 2*(deltax*u)*(deltax*un)) + eta*(Ax*u)**2*(Ax*un))
        u = un
    print('The LEP scheme used {:f} seconds.'.format(time.time()-tic))
    uNs[1,:] = u
    plt.plot(x,u,'--',color=(.6,.8,1))
    
    # LIGEP
    u = u0.copy()
    tic = time.time()
    deltacx = .5/h*spdiags([e,-e,e,-e], np.array([-M+1,-1,1,M-1]), M, M)
    #D1 = deltacx.toarray()
    D1 = fourierD1(M,P)
    for i in range(N):
        A = 1/k*I + .5*eta*D1*spdiags(u,0,M,M) + .5*gamma**2*np.matmul(D1,np.matmul(D1,D1))
        b = 1/k*u-.5*gamma**2*np.dot(np.matmul(D1,np.matmul(D1,D1)),u)
        un = la.solve(A,b)
        energies[i] = h*1/6*sum(-gamma**2*3*(deltax*u)**2 + eta*(Ax*u)**3)
        energiesK[i] = h*1/6*sum(-gamma**2*((deltax*u)**2 + 2*(deltax*u)*(deltax*un)) + eta*(Ax*u)**2*(Ax*un))
        u = un
    print('The LIGEP scheme used {:f} seconds.'.format(time.time()-tic))
    uNs[2,:] = u
    plt.plot(x,u,color=(.8,0,.2))
    
    # GEP
    u = u0.copy()
    tic = time.time()
    for i in range(N):
        un = newton_KdV_GEP(u,u,eta,gamma,D1,I,k,M,1e-10)
        energies[i] = h*1/6*sum(-gamma**2*3*(deltax*u)**2 + eta*(Ax*u)**3)
        energiesK[i] = h*1/6*sum(-gamma**2*((deltax*u)**2 + 2*(deltax*u)*(deltax*un)) + eta*(Ax*u)**2*(Ax*un))
        u = un
    print('The GEP scheme used {:f} seconds.'.format(time.time()-tic))
    uNs[3,:] = u
    plt.plot(x,u,'--',color=(1,.6,.7))
    plt.legend(['LILEP', 'LEP', 'LIGEP', 'GEP'])
    plt.show()