import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf 
import statsmodels.api as sm
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# Reading CSV file
startups_df = pd.read_csv("../Data/50_Startups.csv")
startups_df.columns = ["RD", "Adm", "MS", "State", "Profit"]
print(startups_df.head())

#EDA and PreProcessing
sns.pairplot(startups_df)
plt.show()

# Creating dummy variables for State Column
# dummy = pd.get_dummies(startups_df["State"], drop_first= False)
# print (dummy.head())
# input()
# data_frame = pd.concat([startups_df, dummy], axis = 1)
# print (data_frame.head())
# data_frame.columns = ["RD", "Adm", "MS", "State", "Profit", "California", "Florida", "NewYork"]

# LabelEncoder for catogorical column "State"
labelencoder = LabelEncoder()
startups_df["State_Label"] = labelencoder.fit_transform(startups_df['State'])
print (startups_df.head())
input()


# Correlation Matrix
df = startups_df.corr()
# print (df)



# Multi Linear Regression model using all features
model1 = smf.ols('Profit~RD+Adm+MS+State_Label',data=startups_df).fit() #>> R^2 = 0.951, P Value for Adm = 0.606, P Value for MS = 0.109, P Value for State_Label = 0.989
# Printing summary of Model1
print (model1.summary())
model1_pred = model1.predict(startups_df)
rmse = model1_pred - startups_df.Profit
print (np.sqrt(np.mean(rmse*rmse)))  #>> RMSE = 8855.325573724913
# Printing coefficients of variables
# print (model1.params)
input()

# Visualization of model1
sm.graphics.plot_partregress_grid(model1) 
plt.show()
input()



model2 = smf.ols('Profit~RD+Adm+MS',data=startups_df).fit() #>> R^2 = 0.951, P Value for Adm = 0.602, P Value for MS = 0.105
# Printing summary of Model1
print (model2.summary())
model2_pred = model2.predict(startups_df)
rmse = model2_pred - startups_df.Profit
print (np.sqrt(np.mean(rmse*rmse)))  #>> RMSE = 8855.34448901514

#Visualization of model2
sm.graphics.plot_partregress_grid(model2) 
plt.show()
input()

# Plotting infelencer plot to find the influencer in the given dataset
sm.graphics.influence_plot(model2)
plt.show()

# Creating new dataframe(startups_newdf) by dropping few rows in the data set that are influencing
startups_newdf = startups_df.drop(startups_df.index[[19,45,46,47,48,49]],axis=0)

# Multi Linear Regression Model
model3 = smf.ols('Profit~RD+Adm+MS',data=startups_newdf).fit() #>> R^2 = 0.961, P Value for Adm = 0.254, P Value for MS = 0.039
print (model3.summary())
model3_pred = model3.predict(startups_newdf)
rmse = model3_pred - startups_newdf.Profit
print (np.sqrt(np.mean(rmse*rmse)))  #>> RMSE = 6729.20632982119
input()

# Visualization of model3
sm.graphics.plot_partregress_grid(model3) 
plt.show()
input()

model4 = smf.ols('Profit~RD+MS',data=startups_newdf).fit() #>> R^2 = 0.959
print (model4.summary())
model4_pred = model4.predict(startups_newdf)
rmse = model4_pred - startups_newdf.Profit
print (np.sqrt(np.mean(rmse*rmse)))  #>> RMSE = 6841.13836136014
input()

# Visualization of model4
sm.graphics.plot_partregress_grid(model4) 
plt.show()
input()

model5 = smf.ols('Profit~Adm+MS',data=startups_newdf).fit() #>> R^2 = 0.651, P Value for Adm=0.033 (Dont Consider this model)
print (model5.summary())
model5_pred = model5.predict(startups_newdf)
rmse = model5_pred - startups_newdf.Profit
print (np.sqrt(np.mean(rmse*rmse)))  #>> 20027.40738050953
input()

# Visualization of model5
sm.graphics.plot_partregress_grid(model5) 
plt.show()
input()

model6= smf.ols('Profit~Adm+RD',data=startups_newdf).fit() #>> R^2 = 0.956, P Value for Adm = 0.055
print (model6.summary())
model6_pred = model6.predict(startups_newdf)
rmse = model6_pred - startups_newdf.Profit
print (np.sqrt(np.mean(rmse*rmse)))   #>> RMSE = 7102.153142209682
input()

# Visualization of model6
sm.graphics.plot_partregress_grid(model6) 
plt.show()
input()


# rsq_adm = smf.ols('Adm~RD+MS',data=startups_df).fit().rsquared
# vif_adm = 1/(1-rsq_adm)
# print (vif_adm)

# rsq_rd = smf.ols('RD~Adm+MS',data=startups_newdf).fit().rsquared  
# vif_rd = 1/(1-rsq_rd)
# print (vif_rd)

# rsq_ms = smf.ols('MS~RD+Adm',data=startups_newdf).fit().rsquared  
# vif_ms = 1/(1-rsq_ms)
# print (vif_ms)



#Final_Model
final_model = smf.ols('Profit~RD+MS',data=startups_newdf).fit()  #>> R^2 = 0.959
print (final_model.summary())
final_pred = final_model.predict(startups_newdf)
# print (final_pred)
rmse = final_pred - startups_newdf.Profit
print (np.sqrt(np.mean(rmse*rmse)))   #>> RMSE = 6841.13836136014
input()


sm.graphics.plot_partregress_grid(final_model) 
plt.show()
