// see the basic linear regression for notes on where I drew my inspiration

Created on Tue Sep 13 11:14:25 2016

@author: ychennay
"""
import pandas as pd
import numpy as np

importfilepath = "C:/Users/ychennay/Documents/Python Scripts/Toy Data.csv"

data = pd.read_csv(importfilepath)

data['Intercept'] = 1
y = data.iloc[:,0]

X= data.iloc[:,1:]
X= np.array(X)
y = np.array(y)
coefficients =data.columns
#%%
def cost_function(X, y, beta):
    m = len(y)
    J = np.sum((X.dot(beta)-y)**2)/2/m
    return J

iterations = 10000
learningrate = .0001

beta = np.zeros(len(X[0,:]))
#%%
cost_history = [0] * iterations
m = len(y)
for i in range(iterations):
    hypothesis=X.dot(beta)
    loss = hypothesis - y
    gradient = X.T.dot(loss)/m
    beta = beta.T - learningrate * gradient
    cost= cost_function(X,y,beta)
    cost_history[i] = cost    
    i = i+1

#display the first 50 results
for i in range(5):
    print("Cost", cost_history[i])    

#print final coefficients

for i in range(len(coefficients)-1):
    print(coefficients[i+1])
    print(beta[i])
