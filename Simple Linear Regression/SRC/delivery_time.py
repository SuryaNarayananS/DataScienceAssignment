import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm


# Reading CSV file
delivery_time = pd.read_csv("../Data/delivery_time.csv")
print (delivery_time.head())

# Exploratory Data Analysis (bar plot of the data set)
X = delivery_time["SortingTime"]
Y = delivery_time["DeliveryTime"]
plt.bar(X, Y)
plt.title("Bar Plot for Sorting Time v/s Delivery Time")
plt.xlabel("SortingTime")
plt.ylabel("DeliveryTime")
plt.show()

#boxplot to check for outliers
plt.boxplot(delivery_time)
plt.title("Box Plot for Sorting Time v/s Delivery Time")
plt.xlabel("SortingTime")
plt.ylabel("DeliveryTime")
plt.show()


# Simple_Linear_Regression model using OLS method
model1 = smf.ols("DeliveryTime~SortingTime",data=delivery_time).fit()
print (model1.summary())  # R^2=0.682, P=0.001
pred = model1.predict(delivery_time.iloc[:])
# print (pred)
# input()

# scatter plot for model1
plt.scatter(x=delivery_time["SortingTime"], y=delivery_time["DeliveryTime"], color="red");
plt.plot(delivery_time['SortingTime'],pred,color='black');
plt.title("Sorting Time v/s Delivery Time")
plt.xlabel('Sorting Time');
plt.ylabel('Delivery Time');
plt.show()
input()

# graph to find out the ifluence data point in the dataset
sm.graphics.influence_plot(model1)
plt.show()
input()

# creating a new dataframe(df) by dropping the influencing data points
df = delivery_time.drop([20,4])
# print (df.head())

# Model2
model2 = smf.ols("DeliveryTime~SortingTime",data=df).fit()
print (model2.summary())  # R^2=0.777, P=0.000
pred = model2.predict(df.iloc[:])

# scatter plot for model2
plt.scatter(x=df["SortingTime"], y=df["DeliveryTime"], color="red");
plt.plot(df['SortingTime'],pred,color='black');
plt.title("Sorting Time v/s Delivery Time")
plt.xlabel('Sorting Time');
plt.ylabel('Delivery Time');
plt.show()
input()

# graph to find out the ifluence data point in the dataset
sm.graphics.influence_plot(model2)
plt.show()
input()

# creating a new dataframe(df1) by dropping the influencing data points
df1 = delivery_time.drop([2,8])
# print (df.head())

# Model3
model3 = smf.ols("DeliveryTime~SortingTime",data=df1).fit()
print (model3.summary())  # R^2=0.758, P=0.003
pred = model3.predict(df1.iloc[:])

# scatter plot for model3
plt.scatter(x=df1["SortingTime"], y=df1["DeliveryTime"], color="red");
plt.plot(df1['SortingTime'],pred,color='black');
plt.title("Sorting Time v/s Delivery Time")
plt.xlabel('Sorting Time');
plt.ylabel('Delivery Time');
plt.show()
input()


# Correlation
correlation1 = pred.corr(delivery_time.DeliveryTime)
correlation2 = pred.corr(df.DeliveryTime)
correlation3 = pred.corr(df1.DeliveryTime)
print (correlation1)   #0.8703856691019976
print (correlation2)   #0.9331951416041506
print (correlation3)   #0.8703856691019976
input()



# Final_Model
Final_Model = smf.ols("DeliveryTime~SortingTime",data=df).fit()
print (Final_Model.summary()) # R^2=0.777, P=0.000
pred = Final_Model.predict(df.iloc[:])
# print (pred)

