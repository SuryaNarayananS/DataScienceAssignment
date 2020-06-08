import pandas as pd
import numpy as np
import scipy 
from scipy import stats
import statsmodels.stats.descriptivestats as sd
from statsmodels.stats.proportion import proportions_ztest

####### Hypothesis Testing using 2-Proportion Test  #######

df=pd.read_csv("../Data/Faltoons.csv")
count=pd.crosstab(df["Weekdays"],df["Weekend"])
print (count)

tab = df.groupby(['Weekdays', 'Weekend']).size()
print (tab)

Weekdays = np.array([113, 287])    # Male and Female walking into the store on weekdays
Weekend = np.array([167, 233])     # Male and Female walking into the store on weekend

stat, pval = proportions_ztest(Weekdays, Weekend,alternative='two-sided') 
print('{0:0.3f}'.format(pval))

stat, pval = proportions_ztest(Weekdays, Weekend,alternative='larger')
print('{0:0.3f}'.format(pval))

if pval<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis") 