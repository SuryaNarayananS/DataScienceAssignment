import pandas as pd
import matplotlib.pylab as plt
from scipy.spatial.distance import cdist 
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats


# Reading CSV file
toyota_df = pd.read_csv("../Data/ToyotaCorolla.csv", engine ='python')
print (toyota_df.head())

#Feature Engineering
toyota_newdf = toyota_df.drop(columns=['Id', 'Model', 'Mfg_Month', 'Mfg_Year',
       'Fuel_Type', 'Met_Color', 'Color', 'Automatic',
       'Cylinders', 'Mfr_Guarantee',
       'BOVAG_Guarantee', 'Guarantee_Period', 'ABS', 'Airbag_1', 'Airbag_2',
       'Airco', 'Automatic_airco', 'Boardcomputer', 'CD_Player',
       'Central_Lock', 'Powered_Windows', 'Power_Steering', 'Radio',
       'Mistlamps', 'Sport_Model', 'Backseat_Divider', 'Metallic_Rim',
       'Radio_cassette', 'Tow_Bar'])
print (toyota_newdf)

#model1
model1 = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota_newdf).fit() #>> R^2=0.864, P Value for CC = 0.179, P Value for Doors = 0.968
print (model1.summary())
prediction = model1.predict(toyota_newdf) 
rmse = prediction - toyota_newdf.Price
print (np.sqrt(np.mean(rmse*rmse)))       #>> RMSE= 1338.2584236201512
input()

#Vizualization for model1
sm.graphics.plot_partregress_grid(model1)
# plt.show()
# input()

#Using Z-score function defined in scipy library to detect the outliers.
z_scores = stats.zscore(toyota_newdf)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
toyota_df_new = toyota_newdf[filtered_entries]
print(toyota_df_new)

#model2
model2 = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota_df_new).fit() #>> R^2=0.857, P Value for Quarterly_Tax = 0.547
print (model2.summary())
prediction = model2.predict(toyota_df_new) 
rmse = prediction - toyota_df_new.Price
print (np.sqrt(np.mean(rmse*rmse)))       #>> RMSE= 1150.16847627202
input()

#Vizualization for model2
sm.graphics.plot_partregress_grid(model2)
# plt.show()
# input()

#Plotting infelencer plot to find the influencer in the given dataset
sm.graphics.influence_plot(model2)
plt.show()

#Creating new dataframe(toyota_df_new1) by dropping few rows in the data set that are influencing
toyota_df_new1 = toyota_df_new.drop(toyota_df_new.index[[523,1058,191,192,189,402,393]],axis=0)

#model3
model3 = smf.ols('Price~Age_08_04+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota_df_new1).fit() #>> R^2=0.837, P Value for Quarterly_Tax = 0.571
print (model3.summary())
prediction = model3.predict(toyota_df_new1) 
rmse = prediction - toyota_df_new1.Price
print (np.sqrt(np.mean(rmse*rmse)))    #>> RMSE=1229.8852312547665
input()

#Vizualization for model3
sm.graphics.plot_partregress_grid(model3)
# plt.show()
# input()


#final_model
final_model = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota_df_new).fit() #>> R^2=0.857, P Value for Quarterly_Tax = 0.547
print (final_model.summary())
prediction = final_model.predict(toyota_df_new) 
rmse = prediction - toyota_df_new.Price
print (np.sqrt(np.mean(rmse*rmse)))       #>> RMSE= 1150.16847627202

sm.graphics.plot_partregress_grid(final_model) 
plt.show()