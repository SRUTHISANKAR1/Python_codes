#####################FORECASTING###################
##################COCOCOLA_SALES##################
#Forecast the CocaCola prices

##Load the libraries
import pandas as pd
import numpy as np

#Load the dataset
cococola_df = pd.read_excel("C:/Users/server/Downloads/CocaCola_Sales_Rawdata.xlsx")
cococola_df.head()
cococola_df.dtypes

#Visualization
import matplotlib.pyplot as plt
cococola_df.plot()
#in  plot we can see that there is a linear upword trend and multiplicative seasonality present

quarter=['Q1','Q2','Q3','Q4']
p=cococola_df["Quarter"][0]
p[0:2]
cococola_df['quarter']=0
for i in range(42):
    p=cococola_df["Quarter"][i]
    cococola_df['quarter'][i]=p[0:2]
    
cococola_df

#creating 12 dummy variables
quarter_dummies = pd.DataFrame(pd.get_dummies(cococola_df['quarter']))
cococola_data = pd.concat([cococola_df,quarter_dummies],axis = 1)
cococola_data.head()

cococola_data.columns  # Index(['Quarter', 'Sales', 'quarter', 'Q1', 'Q2', 'Q3', 'Q4'], dtype='object')

#input t,t_squared,log_Sales
cococola_data["t"] = np.arange(1,43)
cococola_data["t_squared"] = cococola_data["t"]*cococola_data["t"]
cococola_data["log_Sales"] = np.log(cococola_data["Sales"])
cococola_data.head()

cococola_data.columns 

#####spli the data into train and test
Train = cococola_data.head(29)
Test = cococola_data.tail(12)

Train.shape, Test.shape  #((29, 10), (12, 10))

####Model Building
#####Create different models and choose best model with least error rate 
import statsmodels.formula.api as smf 

####LINEAR MODEL
linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear   #808.707372465433

###EXPONENTIAL MODEL
Exp = smf.ols('log_Sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp  #638.2398884287844

#QUADRATIC MODEL
Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad   # 522.3073674097535 

# ADDITIVE SEASONALITY
add_sea = smf.ols('Sales~Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[["Q1","Q2","Q3","Q4"]]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea   #1825.6996837161034

#ADDITIVE SEASONALITY WITH QUADRATIC
add_sea_Quad = smf.ols('Sales~t+t_squared+Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[["Q1","Q2","Q3","Q4",'t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad ###593.2706704101529

#MULTIPLICATIVE SEASONALITY
Mul_sea = smf.ols('log_Sales~Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea   # 1869.6395085708914

#Table representations
data = {"MODEL":pd.Series(["linear_model","Exp","Quad ","add_sea","add_sea_Quad","Mul_sea"]),"rmse_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
#MODEL  rmse_Values
#0  linear_model   808.707372
#1           Exp   638.239888
#2         Quad    522.307367  # #Least Error model
#3       add_sea  1825.699684
#4  add_sea_Quad   593.270670
#5       Mul_sea  1869.639509

#Combining Training & test data to build Additive seasonality using Quadratic Trend
cococola_data.head()

#######Prediction
#split the data into train and test
train_set = cococola_data.sample(frac=0.75, random_state=0)
test_set = cococola_data.drop(train_set.index)    
train_set.shape,test_set.shape  #((32, 10), (10, 10))
test_set

#use "Quad" model  for predicting future sales
final_model= smf.ols('Sales~t+t_squared',data=train_set).fit()
final_model.summary()

pred_new_Sales  = pd.Series(final_model.predict(test_set))
pred_new_Sales

# Compare with original value and predicted values
pred=pd.DataFrame(pred_new_Sales )
future_df=pd.concat([test_set[["Sales"]],pred],axis=1)

future_df.columns=["Actual sales","Forcasting sales"]
future_df


#Visualize actual sales and forecasting sales

future_df[['Actual sales', 'Forcasting sales']].plot(figsize=(12, 8))
