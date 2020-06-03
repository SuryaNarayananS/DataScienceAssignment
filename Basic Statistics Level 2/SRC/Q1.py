import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df= pd.read_excel("Q1.xlsx")
print (df)
df.columns = ["Name_of_company", "Measure_X"]

X=df["Name_of_company"]
y=df["Measure_X"]

plt.boxplot(y)
plt.title("Box Plot for Measure_X")
plt.show()

#calculating mean
print (np.mean(y))   #>>  Mean = 0.3327133333333333

#calculating Variance
print (np.var(y))    #>>  Variance = 0.026800350488888885

#calculating Standard Deviation
print (np.std(y))    #>>   Standard Deviation = 0.16370812590976933

