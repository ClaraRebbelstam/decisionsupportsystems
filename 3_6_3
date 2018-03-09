# -*- coding: utf-8 -*-
"""
Decision Sypport Systems 
3.6.2

Kasper Kronborg Larsen
"""
import pandas as pd
import statsmodels.api as sm

data = pd.read_csv(open("../Desktop/DecisionSupportSystems/Exercise/Boston.csv"))

    
y = data['medv']
X = data[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']]

X = sm.add_constant(X) ## let's add an intercept to our model

model = sm.OLS(y, X).fit() ## sm.OLS(output, input)

# Print out the statistics
model.summary()


