import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima 

  
# Read the AirPassengers dataset 
airline = pd.read_excel('../Data/Airlines+Data.xlsx', sheet_name="Sheet1", index_col ='Month', parse_dates = True) 
print (airline.head())
  
# ETS Decomposition 
result = seasonal_decompose(airline['Passengers'], model ='multiplicative') 
result1 = seasonal_decompose(airline['Passengers'], model ='Additive') 

# ETS plot  
result.plot()
plt.show()
result1.plot()
plt.show()

# Ignore harmless warnings 
import warnings 
warnings.filterwarnings("ignore") 
  
# Fit auto_arima function to AirPassengers dataset 
stepwise_fit = auto_arima(airline['Passengers'], start_p = 1, start_q = 1, 
                          max_p = 3, max_q = 3, m = 12, 
                          start_P = 0, seasonal = True, 
                          d = None, D = 1, trace = True, 
                          error_action ='ignore',
                          suppress_warnings = True,
                          stepwise = True)         
  
print (stepwise_fit.summary())


# Split data into train / test sets 
train = airline.iloc[:len(airline)-12]
test = airline.iloc[len(airline)-12:] # set one year(12 months) for testing 
  
# Fit a SARIMAX(1, 1, 0)x(1, 1, 0, 12) on the training set 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
  
model = SARIMAX(train['Passengers'],  
                order = (1, 1, 0),  
                seasonal_order =(1, 1, 0, 12)) 
  
result = model.fit() 
print(result.summary())


start = len(train) 
end = len(train) + len(test) - 1
  
# Predictions for one-year against the test set 
predictions = result.predict(start, end, 
                             typ = 'levels').rename("Predictions") 
  
# plot predictions and actual values 
predictions.plot(legend = True) 
test['Passengers'].plot(legend = True)

# Load specific evaluation tools 
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse 
  
# Calculate root mean squared error 
rmse(test["Passengers"], predictions)
  
# Calculate mean squared error 
print(mean_squared_error(test["Passengers"], predictions))   #>> RMSE = 106.74352828969727