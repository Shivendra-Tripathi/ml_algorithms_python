import numpy as np
import math

#Function that performs gda and returns the results
def gda_binary_learn(x,y):
    
    n,m=x[0].size,y.size           #Stores the dim of a and Sample size
    m0,m1=0,0                   #Stores the size of class "0" samples and class "1" samples
    phi=0                       #Stores the P(Y=1)
    meu_0=0                     #Stores the mean of the class "0"
    meu_1=0                     #Stores the mean of the class "1"
    covar_mat=np.zeros((n,n))   #Covariance matrix
    
    #Calculating the m0 and m1
    m0=np.sum(y==0)
    m1=m-m0
    
    #Calculating the phi
    phi=m1/m
    
    #Calculating the meu0 and meu1 
    meu_0=np.mean(x[y==0],axis=0)
    meu_1=np.mean(x[y==1],axis=0)
    
    #Calculating the covariance matrix
    diff = x - meu_0
    diff[y==1] = x[y==1] - meu_1
    covar_mat = (diff.T @ diff) / m

    
    #Returning the parameters 
    #The first element of tuple is for plotting purposes
    #Second element is for prediction purposes
    return ( (phi , meu_0 , meu_1 , covar_mat) ,(phi , meu_0 , meu_1 , np.linalg.inv(covar_mat) , np.linalg.det(covar_mat)) )


#Function to predict the Outcomes
def gda_binary_predict(x,params):
    
    if np.isclose(params[4],0):
        raise ValueError("Covariance(params[4]) Matrix has det=0")
    n=x.size
    lg_c=-0.5*(n*np.log(2*np.pi) + np.log(params[4]))
    diff_0,diff_1=x-params[1],x-params[2]
       
    lg_p_x_y0 = lg_c - 0.5*(diff_0.T @ params[3] @ diff_0)
    lg_p_x_y1=lg_c - 0.5*(diff_1.T @ params[3] @ diff_1)
    lg_p_y1=np.log(params[0])
    lg_p_y0=np.log((1-params[0]))
    
    if lg_p_x_y0+lg_p_y0 > lg_p_x_y1+lg_p_y1:
        return "0"
    elif lg_p_x_y0+lg_p_y0 < lg_p_x_y1+lg_p_y1:
        return "1"
    return "="                                 # "=" when both probabilities are equal
    

    
    
    
    
    
    
    