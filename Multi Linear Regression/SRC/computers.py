import pandas as pd
import matplotlib.pylab as plt
from scipy.spatial.distance import cdist 
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

#reading CSV file
comp_df = pd.read_csv("../Data/Computer_Data.csv")

#PreProcessing data
# Dropping first column
comp_newdf = comp_df.drop(columns="No")
# print (comp_newdf.head())

#Replacing yes and no with 1 and 0
comp_newdf["cd"]= comp_newdf["cd"].replace(["no", "yes"],[0,1])
comp_newdf["multi"]= comp_newdf["multi"].replace(["no", "yes"],[0,1])
comp_newdf["premium"]= comp_newdf["premium"].replace(["no", "yes"],[0,1])

# print (comp_newdf["cd"])
# print (comp_newdf["multi"])
# print (comp_newdf["premium"])
print (comp_newdf.head())
# input()

#Correlation Matrix
df = comp_newdf.corr()
print (df)
input()

#Checking for Missing_Value
missing_value = comp_newdf.isnull().values.any()
print (missing_value)

#Multi Linear Regression model using all features
model1 = smf.ols('price~speed+hd+ram+screen+cd+multi+premium+ads+trend',data=comp_newdf).fit() #>>R^2=0.776, P=0.000
print (model1.summary())
prediction = model1.predict(comp_newdf)
rmse = prediction - comp_newdf.price
print (np.sqrt(np.mean(rmse*rmse)))    #>> RMSE=275.1298188638718
input()

#Vizualization for model1
sm.graphics.plot_partregress_grid(model1)
# plt.show()
# input()

#Plotting infelencer plot to find the influencer in the given dataset
sm.graphics.influence_plot(model1)
plt.show()

#Creating new dataframe(comp_dataframe) by dropping few rows in the data set that are influencing
comp_dataframe = comp_newdf.drop(comp_newdf.index[[1440,1700,5960,4477,900,101,3783]],axis=0)

#model2
model2 = smf.ols('price~speed+hd+ram+screen+cd+multi+premium+ads+trend',data=comp_dataframe).fit() #>> R^2=0.777 P=0.000
print (model2.summary())
prediction = model2.predict(comp_dataframe)
rmse = prediction - comp_dataframe.price
print (np.sqrt(np.mean(rmse*rmse))) #>> RMSE=272.7510599483028
input()

#Vizualization for model2
sm.graphics.plot_partregress_grid(model2)
# plt.show()
# input()


#Using Z-score function defined in scipy library to detect the outliers.
z_scores = stats.zscore(comp_newdf)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
new_df = comp_newdf[filtered_entries]
print(new_df)


#model3
model3 = smf.ols('price~speed+hd+ram+screen+cd+multi+premium+ads+trend',data=new_df).fit() #>> R^2=0.806 P=0.000
print (model3.summary())
prediction = model3.predict(new_df)
rmse = prediction - new_df.price
print (np.sqrt(np.mean(rmse*rmse)))    #>> RMSE=238.72842758593865
input()

#Vizualization for model3
sm.graphics.plot_partregress_grid(model3)
# plt.show()
# input()


#model4
model4 = smf.ols('price~speed+hd+ram+screen+cd+premium+trend',data=new_df).fit() #>> R^2=0.799, P=0.000
print (model4.summary())
prediction = model4.predict(new_df)
rmse = prediction - new_df.price
print (np.sqrt(np.mean(rmse*rmse)))    #>> RMSE=242.7008421

input()

#Vizualization for model4
sm.graphics.plot_partregress_grid(model4)
# plt.show()
# input()


#model5
model5 = smf.ols('price~speed+hd+ram+screen+cd+ads',data=new_df).fit() #>> R^2=0.587, P=0.000
print (model5.summary())
prediction = model5.predict(new_df)
rmse = prediction - new_df.price
print (np.sqrt(np.mean(rmse*rmse)))    #>> RMSE=238.72842758593865
input()

#Vizualization for model5
sm.graphics.plot_partregress_grid(model5)
# plt.show()
# input()



#final_model
final_model = smf.ols('price~speed+hd+ram+screen+cd+multi+premium+ads+trend',data=new_df).fit() #>> R^2=0.806 P=0.000
print (final_model.summary())
prediction = final_model.predict(new_df)
rmse = prediction - new_df.price
print (np.sqrt(np.mean(rmse*rmse)))    #>> RMSE=238.72842758593865
input()

#Vizualization for final_model
sm.graphics.plot_partregress_grid(final_model)
plt.show()
