import numpy as np
import quadprog
from pinocchio import skew
from numpy.linalg import norm,inv,pinv,svd,eig

def contactForceDistribution(phi,footShape):
    '''
    Distribute a central 6d force into N 3d forces exerted at each foot corner,
    given the shape of the foot.

    min_fk    || sum(f_k) - phi[:3] ||**2 + || sum(pk x fk) - phi[3:] ||**2
    st  f_k in K  ie f_kz**2 >= mu (f_xz**2 + f_yz**2) 
    '''
    
    mu = 1.
    K3 = np.array([
        [  mu,  0,1 ],
        [ -mu,  0,1 ],
        [   0, mu,1 ],
        [   0,-mu,1 ]
    ])
    K = np.block([
        [ K3*(1 if c==r else 0) for c,__c in enumerate(footShape) ]
        for r,__r in enumerate(footShape) ])
    k = np.zeros(K.shape[0])

    reg = 1e-9 # Regularization, because quadprog wants a >0 hessian
    H = np.eye(footShape.shape[0]*3)*reg
    g = np.zeros(footShape.shape[0]*3)*reg
    
    '''
    || sum(f_k) - f ||**2 = 2 f.T S [f_1..f_n]  + [f_1..f_n].T S.T S [f_1..f_n] 
    with S = [I_3 ... I_3]
    '''
    
    S = np.block([np.eye(3) for _ in range(4) ])
    H += S.T@S
    g += S.T@phi[:3]
    
    '''
    || sum(p_k x f_k) - tau ||**2 = 2*tau.T P [f_1..f_n] + [f_1..f_n].T P.T P [f_1..f_n]
    with P = [p_1x ... p_nx]
    '''
    P = np.block([ skew(p_k) for p_k in footShape ])
    H += P.T@P
    g += P.T@phi[3:]
    
    sol,f,xu,it,lag,iact = quadprog.solve_qp(H,g,K.T,k,0)

    return np.split(sol,4)


class ContactForceDistributor:
    '''
    Distribute a central 6d force into N 3d forces exerted at each foot corner,
    given the shape of the foot.
        
    min_fk    || sum(f_k) - phi[:3] ||**2 + || sum(pk x fk) - phi[3:] ||**2
    st  f_k in K  ie f_kz**2 >= mu (f_xz**2 + f_yz**2) 
    '''
    def __init__(self,footShape,mu=1.):
        self.footShape = footShape
        self.mu = mu
        
        K3 = np.array([
            [  mu,  0,1 ],
            [ -mu,  0,1 ],
            [   0, mu,1 ],
            [   0,-mu,1 ]
        ])
        K = np.block([
            [ K3*(1 if c==r else 0) for c,__c in enumerate(footShape) ]
            for r,__r in enumerate(footShape) ])
        k = np.zeros(K.shape[0])

        reg = 1e-9 # Regularization, because quadprog wants a >0 hessian
        H = np.eye(footShape.shape[0]*3)*reg
        g = np.zeros(footShape.shape[0]*3)
    
        '''
        || sum(f_k) - f ||**2 = 2 f.T S [f_1..f_n]  + [f_1..f_n].T S.T S [f_1..f_n] 
        with S = [I_3 ... I_3]
        '''
    
        S = np.block([np.eye(3) for _ in range(4) ])
        H += S.T@S
        
        '''
        || sum(p_k x f_k) - tau ||**2 = 2*tau.T P [f_1..f_n] + [f_1..f_n].T P.T P [f_1..f_n]
        with P = [p_1x ... p_nx]
        '''
        P = np.block([ skew(p_k) for p_k in footShape ])
        H += P.T@P

        self.K = K
        self.k = k
        self.P = P
        self.S = S
        self.H = H
        self.g = g

    def solve(self,phi):
        assert(phi.shape==(6,))
        self.g.fill(0)
        self.g += self.S.T@phi[:3]
        self.g += self.P.T@phi[3:]
        sol,f,xu,it,lag,iact = quadprog.solve_qp(self.H,self.g,self.K.T,self.k,0)
        return np.split(sol,4)


if __name__ == "__main__":
    print('test')
    size = 0.122, 0.205
    footShape = np.array( [
        [ size[0]/2, size[1]/2,0],
        [-size[0]/2, size[1]/2,0],
        [-size[0]/2,-size[1]/2,0],
        [ size[0]/2,-size[1]/2,0] ])

    phi = np.random.rand(6)*2-1
    phi[2] = abs(phi[2])
    phi = phi*(1,1,100,.1,.1,1)
    phi[:] = [0,0,100,0,5,0]

    contactForceDistribution = ContactForceDistributor(footShape,mu=1.)
    fs = contactForceDistribution.solve(phi)

    assert( norm(sum(fs)-phi[:3]) < 1e-6 )
    assert( norm(sum([ np.cross(pk,fk) for pk,fk in zip(footShape,fs) ]) - phi[3:]) < 1e-6 )
    
