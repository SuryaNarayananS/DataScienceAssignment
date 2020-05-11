import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Reading CSV file
emp_data = pd.read_csv("../Data/emp_data.csv")
print (emp_data)

# Exploratory Data Analysis (bar plot of the data set)
# X = emp_data["SH"]
# Y = emp_data["COR"]
# plt.bar(X, Y)
# plt.title("Bar Plot for SH v/s COR")
# plt.xlabel("SH")
# plt.ylabel("COR")
# plt.show()

#boxplot to check for outliers
plt.boxplot(emp_data)
plt.title("Box Plot for SH v/s COR")
plt.xlabel("SH")
plt.ylabel("COR")
plt.show()

# Simple_Linear_Regression model using OLS method
model1 = smf.ols("COR~SH",data=emp_data).fit()
print (model1.summary()) # R^2=0.831, P=0.000
pred = model1.predict(emp_data.iloc[:])
# print (pred)

# Scatter plot for model1
plt.scatter(x=emp_data["SH"], y=emp_data["COR"], color="red");
plt.plot(emp_data['SH'],pred,color='black');
plt.title("SH v/s COR")
plt.xlabel('SH');
plt.ylabel('COR')
plt.show()
input()

# graph to find out the ifluence data point in the dataset
sm.graphics.influence_plot(model1)
plt.show()
input()

# creating a new dataframe(df) by dropping the influencing data points
df = emp_data.drop([0,9])
# print (df.head())

# Model2
model2 = smf.ols("COR~SH",data=df).fit()
print (model2.summary()) # R^2=0.905, P=0.000
pred = model2.predict(df.iloc[:])
# print (pred)

# Scatter plot for model2
plt.scatter(x=df["SH"], y=df["COR"], color="red");
plt.plot(df['SH'],pred,color='black');
plt.title("SH v/s COR")
plt.xlabel('SH');
plt.ylabel('COR')
plt.show()
input()

# graph to find out the ifluence data point in the dataset
sm.graphics.influence_plot(model2)
plt.show()
input()

# creating a new dataframe(df1) by dropping the influencing data points
df1 = emp_data.drop([1,8])
# print (df.head())

# Model3
model3 = smf.ols("COR~SH",data=df1).fit()
print (model3.summary()) # R^2=0.783, P=0.001
pred = model3.predict(df1.iloc[:])
# print (pred)

# Scatter plot for model3
plt.scatter(x=df1["SH"], y=df1["COR"], color="red");
plt.plot(df1['SH'],pred,color='black');
plt.title("SH v/s COR")
plt.xlabel('SH');
plt.ylabel('COR')
plt.show()
input()

# Correlation
correlation1 = pred.corr(emp_data.COR)
correlation2 = pred.corr(df.COR)
correlation3 = pred.corr(df1.COR)
print (correlation1)   #0.8851088531027236
print (correlation2)   #0.9903227789336354
print (correlation3)   #0.8851088531027236
input()


# Final_Model
Final_Model = smf.ols("COR~SH",data=df).fit()
print (Final_Model.summary()) # R^2=0.905, P=0.000
pred = Final_Model.predict(df.iloc[:])
# print (pred)