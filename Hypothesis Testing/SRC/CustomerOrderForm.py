import pandas as pd
import numpy as np
import scipy 
from scipy import stats
import statsmodels.stats.descriptivestats as sd

####### Hypothesis Testing using chi-square test#######

df=pd.read_csv("../Data/Costomer+OrderForm.csv")
count_1 = df["Phillippines"].value_counts()
print (count_1)
count_2 = df["Indonesia"].value_counts()
print (count_2)
count_3 = df["Malta"].value_counts()
print (count_3)
count_4 = df["India"].value_counts()
print (count_4)

df1= pd.DataFrame()
df1["Phillippines"] = count_1
df1["Indonesia"] = count_2
df1["Malta"] = count_3
df1["India"] = count_4
print (df1)


Chisquares_results=scipy.stats.chi2_contingency(df1)
Chi_pValue=Chisquares_results[1]
print("p-value is: "+str(Chi_pValue))       #>> Chi_pValue = 0.2771020991233135

if Chi_pValue<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")
