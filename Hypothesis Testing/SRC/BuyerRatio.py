import pandas as pd
import numpy as np
import scipy 
from scipy import stats
import statsmodels.stats.descriptivestats as sd

####### Hypothesis Testing using chi-square test#######

df= pd.read_csv("../Data/BuyerRatio.csv")
print (df.head())

tab = df.groupby(['Observed_Values', 'East', 'West', 'North', 'South']).size()
print (tab)

Chisquares_results=scipy.stats.chi2_contingency(tab)
Chi_pValue=Chisquares_results[1]
print("p-value is: "+str(Chi_pValue))

if Chi_pValue<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis") 