This code is based almost entirely on the derivation provided at http://nbviewer.jupyter.org/github/tfolkman/learningwithdata/blob/master/Logistic%20Gradient%20Descent.ipynb.
However, I have tweaked it to include support for a beta_0 intercept value.


import pandas as pd
import numpy as np
import math

importfilepath = "C:/Users/ychennay/Documents/Python Scripts/Binary Classification.csv"
data = pd.read_csv(importfilepath)

y = data.iloc[:,0]
X= data.iloc[:,1:]

X= np.array(X)
y = np.array(y)
coefficients =data.columns[1:]
beta = np.zeros(len(X[0,:])+1)
#%%
def logistic_function(theta, x):
    return float(1) / (1+math.e**(-x.dot(theta)))
    #%%
def logistic_gradient(theta, x, y):
    first_calculation = logistic_function(theta, x) - np.squeeze(y)
    final_calculation = first_calculation.T.dot(x)
    return final_calculation

#%%
def cost_function(theta, x, y):
    logistic_function_v = logistic_function(theta,x)
    step1 = y * np.log(logistic_function_v)
    step2 = (1-y) * np.log(1 - logistic_function_v)
    final = -step1 - step2
    return np.mean(final)
#%%   
    
def gradient_descent(theta_values, X, y, lr= .00001, change_threshold = .001):
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X= np.insert(X, len(X[1,:])-1,1, axis =1) #inserting the intercept betas here
    cost = cost_function(theta_values, X, y)
    i = 1
    change_cost = 1
    
    while (change_cost >= change_threshold):#change_cost > change_threshold):
        old_cost = cost
        theta_values = theta_values - (lr * logistic_gradient(theta_values, X,y))
        cost = cost_function(theta_values, X, y)
        change_cost = old_cost - cost
        i = i + 1
    
    return theta_values
