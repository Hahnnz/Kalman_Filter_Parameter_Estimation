import numpy as np

def Kalman_Filter(Y, U, A, B, C, D, Q, R, X0, P0):
    N = Y.shape[1]
    N_ms, N_st = C.shape
    
    Xp = np.zeros((N_st, N))
    Pp = np.zeros((N_st, N_st, N))
    Xf = np.zeros((N_st, N))
    Pf = np.zeros((N_st, N_st, N))
    Kf = np.zeros((N_st, N_ms))

    if B.size == 0 : B = 0

    #LL = 0
    
    for i in range(N):
        if i == 1 : #initialize
            Xp[:,0] = X0.reshape(-1)
            Pp[:,:,0] = P0
        else :
            Xp[:,i] = A@Xf[:,i-1] + B@U[:,i-1]
            Pp[:,:,i] = A@Pf[:,:,i-1]@A.transpose()+Q
        
        Rei = C@Pp[:,:,i]@C.transpose() + R
        Rei = (Rei.transpose() + Rei)/2
        ReiInv = np.linalg.inv(Rei)
        Kf = Pp[:,:,i]@C.transpose()@ReiInv # for speed up
        
        innov = Y[:,i] - C@Xp[:,i]-D@U[:,i]
        Xf[:,i] = Xp[:,i] + Kf@innov

        #if narout > 5:
        #    LL = LL + np.log(np.linalg.det(Rei)) + innov.transpose()*ReiInv@innov  # for speed up
        #    LL = LL + np.log(abs(np.linalg.det(Rei))) + innov.transpose()/Rei@innov
        
    #LL = -.5*LL
    
    return Xp, Pp, Xf, Pf, Kf