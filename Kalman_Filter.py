import numpy as np
    
class Linear_Kalman_Filter:
    def __init__(self, Y, U, A, B, C, D, Q, R, x_0, P_0):
        self.Y_last = Y
        self.U_last = U
        
        self.A_last = A
        self.B_last = B
        self.C_last = C
        self.D_last = D
        
        self.Q_last = Q
        self.R_last = R
        self.x_last = x_0
        self.P_last = P_0
 
    def update(self, Y=None, U=None, A=None, B=None, C=None, D=None, Q=None, R=None):
        # update old data
        # Consider there is no Prediction values. don't update, just follow dynamics
        
        if not U : U = self.U_last
        else: self.U_last = U
        
        if not A : A = self.A_last
        else: self.A_last = A
        if not B : B = self.B_last
        else: self.B_last = B
        if not C : C = self.C_last
        else: self.C_last = C
        if not D : D = self.D_last
        else: self.D_last = D
            
        if not Q: Q = self.Q_last
        else : self.Q_last = Q
        if not R : R = self.R_last
        else : self.R_last = R
            
        # Run Linear Kalman Filter Algorithm
        x = A@self.x_last + B@self.U_last # predict x
        P = A@self.P_last@A.transpose() + Q # predict p
        # check optimization for this mul

        K_k = P@C.transpose() @ np.linalg.inv(C@P@C.transpose() + R) # calc Kalman gain
        x = x + K_k@(Y - C@x) - self.D_last@self.U_last # calc x
        P = P - K_k@C@P # calc P
        
        # update old data
        self.x_last = x
        self.P_last = P 

        return x
