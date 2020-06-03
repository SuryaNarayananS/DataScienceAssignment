import pandas as pd
import numpy as np
import scipy 
from scipy import stats
import statsmodels.stats.descriptivestats as sd
import statsmodels.formula.api as smf
import statsmodels.api as sm

####### Hypothesis Testing using One - Way Anova #######


df=pd.read_csv("../Data/LabTAT.csv")
print (df.head())
df.columns= ["Lab1", "Lab2", "Lab3", "Lab4"]

#Shapiro Test
Lab1=stats.shapiro(df.Lab1)
Lab1_pValue=Lab1[1]
print("p-value for Lab1 is: "+str(Lab1_pValue))       #>>  p-value for Lab1 is: 0.5506953597068787

Lab2=stats.shapiro(df.Lab2)
Lab2_pValue=Lab2[1]
print("p-value for Lab2 is: "+str(Lab2_pValue))       #>>  p-value for Lab2 is: 0.8637524843215942

Lab3=stats.shapiro(df.Lab3)
Lab3_pValue=Lab3[1]
print("p-value for Lab3 is: "+str(Lab3_pValue))       #>>  p-value for Lab3 is: 0.4205053448677063

Lab4=stats.shapiro(df.Lab4)
Lab4_pValue=Lab4[1]
print("p-value for Lab4 is: "+str(Lab4_pValue))       #>>  p-value for Lab4 is: 0.6618951559066772

#Varience Test 
print (scipy.stats.levene(df.Lab1, df.Lab2))
print (scipy.stats.levene(df.Lab2, df.Lab3))
print (scipy.stats.levene(df.Lab3, df.Lab4))
print (scipy.stats.levene(df.Lab4, df.Lab1))
print (scipy.stats.levene(df.Lab1, df.Lab3))
print (scipy.stats.levene(df.Lab2, df.Lab4))
input()

#One-Way Anova

print (scipy.stats.levene(df.Lab1, df.Lab2, df.Lab3, df.Lab4))    #>> LeveneResult = (statistic=2.599642500418024, pvalue=0.05161343808309816)
print (stats.f_oneway(df.Lab1, df.Lab2, df.Lab3, df.Lab4))        #>> F_onewayResult = (statistic=118.70421654401437, pvalue=2.1156708949992414e-57)
input()

#model
mod=smf.ols('Lab1~Lab2+Lab3+Lab4',data=df).fit()
aov_table=sm.stats.anova_lm(mod,type=2)
print(aov_table)

if Lab1_pValue<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")


if Lab2_pValue<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")


if Lab3_pValue<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis") 


if Lab4_pValue<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")




