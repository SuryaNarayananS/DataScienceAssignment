import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Reading CSV file
cal = pd.read_csv("../Data/calories_consumed.csv")
print (cal.head())

# Exploratory Data Analysis (bar plot of the data set)
# X = cal["CC"]
# Y = cal["WG"]
# plt.bar(X, Y)
# plt.title("Bar Plot for CC v/s WG")
# plt.xlabel("CC")
# plt.ylabel("WG")
# plt.show()

#boxplot to check for outliers
plt.boxplot(cal)
plt.title("Box Plot for CC v/s WG")
plt.xlabel("CC")
plt.ylabel("WG")
plt.show()

# Simple_Linear_Regression model using OLS method
model1 = smf.ols("WG~CC",data=cal).fit()
print (model1.summary())  # R^2=0.897, P=0.000
pred = model1.predict(cal.iloc[:])
# print (pred)

# Scatter plot for model1
plt.scatter(x=cal["CC"], y=cal["WG"], color="red");
plt.plot(cal['CC'],pred,color='black');
plt.title("CC v/s WG")
plt.xlabel('Calories_Consumed');
plt.ylabel('Weight_Gained_in_Grams')
plt.show()
input()

# graph to find out the ifluence data point in the dataset
sm.graphics.influence_plot(model1)
plt.show()
input()

# creating a new dataframe(df) by dropping the influencing data points
df = cal.drop([9])
# print (df.head())

# Model2
model2 = smf.ols("WG~CC",data=df).fit()
print (model2.summary())  # R^2=0.840, P=0.001
pred = model2.predict(df.iloc[:])
# print (pred)

# Scatter plot for model2
plt.scatter(x=df["CC"], y=df["WG"], color="red");
plt.plot(df['CC'],pred,color='black');
plt.title("CC v/s WG")
plt.xlabel('Calories_Consumed');
plt.ylabel('Weight_Gained_in_Grams')
plt.show()
input()

#Correlation
correlation1 = pred.corr(cal.WG)
correlation2 = pred.corr(df.WG)
print (correlation1)   #0.9164959320993801
print (correlation2)   #0.9164959320993801
input()

# Final_Model
Final_Model = smf.ols("WG~CC",data=cal).fit()
print (Final_Model.summary())  # R^2=0.897, P=0.000
pred = Final_Model.predict(cal.iloc[:])
# print (pred)