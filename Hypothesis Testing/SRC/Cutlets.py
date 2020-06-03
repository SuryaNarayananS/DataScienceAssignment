import pandas as pd
import numpy as np
import scipy 
from scipy import stats
import statsmodels.stats.descriptivestats as sd
from scipy.stats import ttest_ind

####### Hypothesis Testing using 2 Sample T test #######


df=pd.read_csv("../Data/Cutlets.csv")
df.columns=["Unit_A", "Unit_B"]
print (df)

Unit_A=stats.shapiro(df.Unit_A)
pValue=Unit_A[1]
print("p-value for Unit_A is: "+str(pValue))   #>> p_value for Unit_A = 0.3199819028377533

Unit_B=stats.shapiro(df.Unit_B)
pValue=Unit_B[1]
print("p-value for Unit_B is: "+str(pValue))    #>> p_value for Unit_B = 0.5224985480308533

print(scipy.stats.anderson(df.Unit_A,dist = 'norm'))  
print(scipy.stats.anderson(df.Unit_B,dist = 'norm'))

#we can proceed with the model 
#Varience test 
print(scipy.stats.levene(df.Unit_A, df.Unit_B))       #>>  LeveneResult = (statistic=0.665089763863238, pvalue=0.4176162212502553)

#2 Sample T test 
print (scipy.stats.ttest_ind(df.Unit_A, df.Unit_B))

print (scipy.stats.ttest_ind(df.Unit_A, df.Unit_B,equal_var = True))    #>>  Ttest_indResult = (statistic=0.7228688704678061, pvalue=0.47223947245995)

if pValue<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis") 