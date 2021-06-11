###########################FORECASTING############################
###########PlasticSales.csv #########################################
###Forecast the dataset
import pandas as pd
import numpy as np
from datetime import datetime,time
import matplotlib.pyplot as plt
import seaborn as sns
plastic_df = pd.read_csv("C:/Users/server/Downloads/PlasticSales.csv")  
plastic_df.head() 
plastic_df.shape  ##60,2

plastic_df.dtypes

# Converting the normal index of plastic_df to time stamp 
plastic_df["Month"]=pd.to_datetime(plastic_df.Month,format="%b-%y")
plastic_df.dtypes
plastic_df.head()

#make Month column as an index
plastic_df.set_index('Month', inplace=True)
plastic_df.index
plastic_df.head()


#check whether any null value present in the data set
plastic_df.isnull().sum()   #no null values

#visualization
plastic_df["Sales"].plot()
plt.show()  #Here sales data following upword trend and multiplicative seasonality

#QQ PLOT
import scipy.stats
import pylab
scipy.stats.probplot(plastic_df.Sales,plot=pylab)
pylab.show()

# Centering moving average for the time series to understand better about the trend character in Walmart
plastic_df.Sales.plot(label="org")
for i in range(2,24,6):
    plastic_df["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(plastic_df.Sales,freq=12)
decompose_ts_add.plot()

# ACF plots and PACF plots on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(plastic_df.Sales,lags=12)
tsa_plots.plot_pacf(plastic_df.Sales,lags=12) # acf and pacf plot showing significance of error

#Split the data into train and test
Train = plastic_df.head(48)
Test = plastic_df.tail(12)
Train.head()
Test.head()


#Two major forecsting methods based on smoothing are
#Moving averages
#Exponential smoothing

#Following are the Exponential smoothing techniques:
#Simple Exponential Method
#Holt method
##Holts winter exponential smoothing with additive seasonality and additive trend
#Holts winter exponential smoothing with multiplicative seasonality and additive trend

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)

# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales"]).fit(smoothing_level=0.2)
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
error1=MAPE(pred_ses,Test.Sales)
error1  # 18.181999865771093

# Holt method 
hw_model = Holt(Train["Sales"]).fit(smoothing_level=0.8, smoothing_slope=0.2)
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
error2=MAPE(pred_hw,Test.Sales)
error2  #23.488809636241623

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=12).fit(smoothing_level=0.6, smoothing_slope=0.15, smoothing_seasonal=0.05)
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
error3=MAPE(pred_hwe_add_add,Test.Sales)
error3   #17.744893029851873

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
error4=MAPE(pred_hwe_mul_add,Test.Sales)
error4   #14.873357409542862  #least error.cosider this model for forcasting

data = {"MODEL":pd.Series(["ses_model","hw_model","hwe_model_add_add","hwe_model_mul_add"]),"MAPE_Values":pd.Series([error1,error2,error3,error4])}
table_MAPE=pd.DataFrame(data)
table_MAPE
# MODEL  MAPE_Values
#0          ses_model    18.182000
#1           hw_model    23.488810
#2  hwe_model_add_add    17.744893
#3  hwe_model_mul_add    14.873357

#display the predicted values
pred_hwe_mul_add 

Test

# Compare with original value and predicted values
pred=pd.DataFrame(pred_hwe_mul_add )
future_df=pd.concat([Test,pred],axis=1)

future_df.columns=["Actual sales","Forcasting sales"]
future_df


##Visualization of actual sales and forecasting sales
future_df[['Actual sales', 'Forcasting sales']].plot(figsize=(12, 8))
