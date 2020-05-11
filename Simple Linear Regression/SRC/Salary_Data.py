import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm


# Reading CSV file
salary_data = pd.read_csv("../Data/Salary_Data.csv")
print (salary_data.head())

#Exploratory Data Analysis (bar plot of the data set)
X = salary_data["YearsExperience"]
Y = salary_data["Salary"]
plt.bar(X, Y)
plt.title("Bar Plot of YearsExperience v/s Salary")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

#boxplot to check for outliers
plt.boxplot(salary_data)
plt.title("Box Plot for YearsExperience v/s Salary")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

# Simple_Linear_Regression model using OLS method
model1 = smf.ols("Salary~YearsExperience",data=salary_data).fit()
print (model1.summary()) # R^2=0.957, P=0.000 
pred = model1.predict(salary_data.iloc[:])
# print (pred)
# input()

# scatter plot for model1
plt.scatter(x=salary_data["YearsExperience"], y=salary_data["Salary"], color="red");
plt.plot(salary_data['YearsExperience'],pred,color='black');
plt.title("YearsExperience v/s Salary")
plt.xlabel('YearsExperience');
plt.ylabel('Salary')
plt.show()
input()

# graph to find out the ifluence data point in the dataset
sm.graphics.influence_plot(model1)
plt.show()
input()

# creating a new dataframe(df) by dropping the influencing data points
df = salary_data.drop([29,19,28])
# print (df.head())

# Model2
model2 = smf.ols("Salary~YearsExperience",data=df).fit()
print (model2.summary()) # R^2=0.953, P=0.000 
pred = model2.predict(df.iloc[:])
# print (pred)

# Scatter plot for Model2
plt.scatter(x=df["YearsExperience"], y=df["Salary"], color="red");
plt.plot(df['YearsExperience'],pred,color='black');
plt.title("YearsExperience v/s Salary")
plt.xlabel('YearsExperience');
plt.ylabel('Salary')
plt.show()
input()

#correlation
correlation1 = pred.corr(salary_data.Salary)
correlation2 = pred.corr(df.Salary)
print (correlation1)  #0.9760347618011087
print (correlation2)  #0.9760347618011087
input()

#Final_Model
Final_Model = smf.ols("Salary~YearsExperience",data=salary_data).fit()
print (Final_Model.summary()) # R^2=0.957, P=0.000 
pred = Final_Model.predict(salary_data.iloc[:])
# print (pred)