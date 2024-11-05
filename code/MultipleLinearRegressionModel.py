#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:46:24 2024

@author: mali
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy,math
np.set_printoptions(precision=2)
from sklearn.linear_model import LinearRegression


df=pd.read_csv('/Users/mali/Downloads/multiple_linear_regression_dataset.csv')

x=df.iloc[:,:2].to_numpy()
y=df.iloc[:,2].to_numpy()

y_train=y.reshape(-1,1)

w_init=np.array([8.9,300])
b_init=25000

x_vec=x[0]


def my_dot(x,w,b):
    m=x.shape[0]
    f_wb=0
    for i in range(m):
        f_wb += x[i]*w[i]
        
    f_wb=f_wb + b
    
    return f_wb

def dot_prod(x,w,b):
    f_wb= np.dot(x,w)+b
    return f_wb

def compute_cost_multi(x,y,w,b):
    m=x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb = (np.dot(x[i],w) + b)
        cost = cost + (f_wb - y[i])**2
    
    total_cost = cost / (2*m)
    
    return total_cost

def compute_gradient_multi(x,y,w,b):
    m,n=x.shape
    dj_dw=np.zeros((n,))
    dj_db=0
    
    for i in range(m):
        f_wb = np.dot(x[i],w) + b
        for j in range(n):
            dj_dw[j] +=  (f_wb - y[i]) * x[i,j]
        dj_db += f_wb - y[i]
        
    dj_dw = dj_dw / m 
    dj_db = dj_db / m
    
    return dj_dw,dj_db

def gradient_descent_multi(x,y,alpha,num_iter,gradient_compute):
    n=x.shape[1]
    b=0
    w=np.zeros((n,))
    
    for i in range(num_iter):
        tmp_w,tmp_b=gradient_compute(x,y,w,b)
        w = w - alpha * tmp_w
        b = b - alpha * tmp_b
        
    return w,b

tmp_alpha = 5.0e-7
iterations = 10000

w_final,b_final = gradient_descent_multi(x,y_train,tmp_alpha,iterations,compute_gradient_multi)

lin_model = LinearRegression()

lin_model.fit(x,y_train)
best_w=lin_model.coef_.reshape(-1,1)
best_b=lin_model.intercept_

pred1=dot_prod(x_vec, w_final, b_final)
pred2=lin_model.predict([x_vec])

## Probably ı can't find correct alpha value so our predictions are very different each other ı will try fix that and ı will share new model