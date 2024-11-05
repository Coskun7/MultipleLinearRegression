#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 12:03:44 2024

@author: mali
"""

import numpy as np
import pandas as pd 
import copy 

df=pd.read_csv('/Users/mali/Downloads/multiple_linear_regression_dataset.csv')

x=df.iloc[:,:2]
y=df.iloc[:,2]

x_train = x.to_numpy()
y_train = y.to_numpy()

print(f'x_train shape {x_train.shape}, type of x_train {type(x_train)}')
print(f'y_train shape {y_train.shape}, type of y_train{type(y_train)}')

w_init=np.array([0.3,30])
b_init=789

print(f'w_init shape {w_init.shape}, type of w_init {type(w_init)}')
print(f'b_init value {b_init}, type of b_init {type(b_init)} ')


#our linear model : w1*x1 + w2*x2 + b


def single_loop_pred(x,w,b):
    """
    

    Parameters
    ----------
    x : (ndarray)
        value of features
    w : (ndarray)
        coefficients
    b : (scalar) model parameter 
        

    Returns
    -------
    f_wb : (scalar) prediction

    """
    ""
    m=x.shape[0]
    f_wb = 0
    for i in range(m):
        f_wb_i = x[i]*w[i]
        f_wb += f_wb_i
    f_wb = f_wb + b
    
    return f_wb

x_ = x.iloc[0,:]
x_vec = x_.to_numpy()
print(f'x_vec shape {x_vec.shape}, type of x_vec {type(x_vec)}')

def dot_pred(x,w,b):
    """
    

    Parameters
    ----------
    x : (ndarray)
        value of features
    w : (ndarray)
        coefficients
    b : (scalar) model parameter 

    Returns
    -------
    f_wb : (scalar) prediction

    """
    f_wb = np.dot(x,w) + b
    
    return f_wb

def compute_cost_function(x,y,w,b):
    """
     

     Parameters
     ----------
     x : (ndarray) shape (m, n) type np.ndarray.
     y : (ndarray) shape (m,) type np.ndarray.
     w : (ndarray) shape (n,) type np.ndarray.
     b : (scalar) type int.

     Returns
     -------
     total_cost : (scalar) type int.

     """ 
    m=x.shape[0]
    cost = 0 
    for i in range(m):
        f_wb = np.dot(x[i],w)+b
        cost += (f_wb -y[i])**2
    total_cost = cost/(2*m)
    return total_cost
cost_first = compute_cost_function(x_train,y_train,w_init,b_init)

def compute_gradient(x,y,w,b):
     m,n = x.shape
     dj_dw=np.zeros((n,))
     dj_db=0
     
     for i in range(m):
         f_wb = np.dot(x[i],w) + b
         for j in range(n):
             dj_dw[j] += (f_wb - y[i])*x[i,j]
         dj_db += f_wb - y[i]
         
     dj_dw = dj_dw / m
     dj_db = dj_db / m
     
     return dj_dw,dj_db
 
    
tmp_w,tmp_b=compute_gradient(x_train,y_train,w_init,b_init)
print(f'tmp_w values {tmp_w}, tmp_b value {tmp_b}')
 
def gradient_descent(x,y,w_in,b_in,alpha,num_iter,gradient_function):
    
    w=copy.deepcopy(w_in)
    b=b_in
    
    for i in range(num_iter):
        dj_dw,dj_db=gradient_function(x,y,w,b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
    return w,b

w_initialize = np.zeros_like(w_init)
b_initialize = 0
iterations = 1000
alpha = 1.0e-9

w_final,b_final = gradient_descent(x_train,y_train,w_initialize,b_initialize,alpha,iterations,compute_gradient)

print(f'w_init : {w_init},w_final {w_final}')
print(f'b_init : {b_init},b_final {b_final}')

cost_final = compute_cost_function(x_train,y_train,w_final,b_final)

print(f'cost_first {cost_first}, cost_final{cost_final}')
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



        
    
