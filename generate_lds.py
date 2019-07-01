import numpy as np

def generate_lds(U, A, B, C, D, Q, R, initX, initV):
    T = U.shape[1]
    os, ss = C.shape
    
    X = np.zeros((ss, T))
    Y = np.zeros((os, T))

    sQ = np.diag(Q)**0.5
    sR = np.diag(R)**0.5
    sQ = sQ.reshape(-1,1)
    sR = sR.reshape(-1,1)
    
    # draw initial state from initX and initV and measure it
    X[:,0] = np.squeeze(initX + (initV**0.5)@np.random.randn(ss,1)) # iniV must be diagonal
    
    if U.size == 0:
        Y[:,0] = C@X[:,0] + np.random.randn(os,1)@sR
    else:
        Y[:,0] = C@X[:,0] + D@U[:,0] + np.random.randn(os,1)@sR
    
    # Another way of initialization
    # draw initial state
    
    if U.size == 0:
        for t in range(1,T):
            X[:,t] = A@X[:,t-1] + sQ*np.random.randn(ss,1)
            Y[:,t] = C@X[:,t] + sR*np.random.randn(os,1)
    else :
        for t in range(1,T):
            noise_Q = (sQ*np.random.randn(ss,1)).reshape(-1,1)
            noise_R = (sR*np.random.randn(os,1)).reshape(-1,1)
            
            xx = (A@X[:,t-1] + B@U[:,t-1]).reshape(-1,1)
            yy = (C@X[:,t] + D@U[:,t]).reshape(-1,1)
            
            X[:,t] = np.squeeze(xx + noise_Q)
            Y[:,t] = np.squeeze(yy + noise_R)
            
    return X, Y