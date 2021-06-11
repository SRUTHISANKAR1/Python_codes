###########################MULTIPLE LINEAR REGRESSION###################################
############################50 startups################

#Prepare a prediction model for profit of 50_startups data.
#Do transformations for getting better predictions of profit and
#make a table containing R^2 value for each prepared model.

#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



##load 50-startups data
startups=pd.read_csv("C:/Users/server/Downloads/50_Startups.csv")
#50 0bs and 5 variables
#Here profit is the target (dependent)variable and 4 independent variables


################EDA###################
startups.head()
startups.tail()
startups.columns
startups.shape


#Data preprocessing
startups.dtypes
#State column is object type convert into number
startups["State"].unique() #to get unique values


#############Dummy variables#######
#here we have 3type of categories Newyork,California,and Florida
#so label encoding  will not work
#use One Hot Encoding to create dummy variables
from sklearn.preprocessing import OneHotEncoder

dummies=pd.get_dummies(startups.State) #Now we got the dummy variable column
dummies.head()
#concatenate dummies with startups datframe
startups_m=pd.concat([startups,dummies],axis="columns")
startups_m.head
#we dont want State column anymore.remove it
#inorder to avoid dummy variable trap( means multi collinearity problem) we can remove first dummy variable column
#take only n-1 dummy variables(if n=4,remove 1 and take only 3)
#use drop method to remove columns and get the final dataframe
final=startups_m.drop(['State','California'],axis='columns')
final.head
final.dtypes
final.columns

final.describe()   #descriptive statistics
final.info()   #showing zero null values
final.isnull().sum()   #No null values

#seggregate the variables into x and y
x=startups.iloc[:,0,1,2,4,5] 
y=startups.iloc[:,3   #4 th variable



## Correlation matrix 
startups.corr()
#R&D spend and profit -good correlation-0.972900
#Marketing spend  and profit -moderately correlated-0.747766
#R&D Spend &Marketing Spend -moderately correlated- 0.724248 (collinearity)

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(startups)


#here outcome is our target variable
#segregate the data into input and output columns
inputv=final.iloc[:,:]
X=inputv.drop('Profit',axis=1)
print(X)
y=final.iloc[:,3]
print(y)
#split the data into train and test
from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=0)  #80%train and 20%test

#fitting multiple linear regression to train dataset
from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(Xtrain,ytrain)   #fitting multiple linear regression to training data
#predict the model using test data
pred=linear_reg.predict(Xtest)
pred
from sklearn import linear_model
ols=linear_model.LinearRegression
model = ols.fit(X, y)
from sklearn.metrics import r2_score
score=r2_score(ytest,pred)
score   #0.9347068473282423   good model
