import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima 

  
# Read the PlasticSales dataset 
plastic_df = pd.read_csv('../Data/PlasticSales.csv',index_col ='Month', parse_dates = True) 
print (plastic_df.head())
  
# ETS Decomposition 
result = seasonal_decompose(plastic_df["Sales"], model ='multiplicative', period=None)
result1 = seasonal_decompose(plastic_df["Sales"], model ='Additive', period=None)

# ETS plot  
result.plot()
plt.show()
result1.plot()
plt.show()

# Ignore harmless warnings 
import warnings 
warnings.filterwarnings("ignore")
  
# Fit auto_arima function to PlasticSales dataset 
stepwise_fit = auto_arima(plastic_df['Sales'], start_p = 1, start_q = 1, 
                          max_p = 3, max_q = 3, m = 12, 
                          start_P = 0, seasonal = True, 
                          d = None, D = 1, trace = True, 
                          error_action ='ignore',
                          suppress_warnings = True,
                          stepwise = True)         
  
print (stepwise_fit.summary())


# Split data into train / test sets 
train = plastic_df.iloc[:len(plastic_df)-12]
test = plastic_df.iloc[len(plastic_df)-12:] # set one year(12 months) for testing 
  
# Fit a SARIMAX(2, 0, 1)x(0, 1, 1, 12) on the training set 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
  
model = SARIMAX(train['Sales'],  
                order = (2, 0, 1),  
                seasonal_order =(0, 1, 1, 12)) 
  
result = model.fit() 
print(result.summary())


start = len(train) 
end = len(train) + len(test) - 1
  
# Predictions for one-year against the test set 
predictions = result.predict(start, end, 
                             typ = 'levels').rename("Predictions") 
  
# plot predictions and actual values 
predictions.plot(legend = True) 
test['Sales'].plot(legend = True)

# Load specific evaluation tools 
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse 
  
# Calculate root mean squared error 
rmse(test["Sales"], predictions)
  
# Calculate mean squared error 
print(mean_squared_error(test["Sales"], predictions))   #>> RMSE = 30981.487578171655