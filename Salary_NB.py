####################NAIVE BAYES#################
###########SALARY TRAIN & TEST DATA####################


####import packages
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


##########Load Salary train Data
Salary_train=pd.read_csv("C:/Users/server/Downloads/SalaryData_Train.csv")  #30161 obs and 14 variables
Salary_train

Salary_test=pd.read_csv("C:/Users/server/Downloads/SalaryData_Test (1).csv")  #15060 obs and 14 variables
Salary_test

#combine two datasets and convert into a single one by using append function
Salary = Salary_train.append(Salary_test)    #45221 obs ,14 variables


#check for null values
pd.isnull(Salary).sum()    #zero null values

colnames=list(Salary.columns)  #to get the column names
colnames



Salary['Salary'].unique()   #to get the unique terms 
#array([' <=50K', ' >50K'], dtype=object)
#convert into numeric format by label encoding

# find numerical variables
numerical = [var for var in Salary.columns if Salary[var].dtype!='O']
print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are :', numerical)


# Seaborn visualization library
import seaborn as sns
# Create the default pairplot
sns.pairplot(Salary)



column=["workclass",'education','educationno','maritalstatus','occupation','relationship','race','sex','native','Salary']

# Import label encoder 
from sklearn import preprocessing 
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
for i in column:
    Salary[i]= label_encoder.fit_transform(Salary[i]) 


Salary['Salary'].unique() 
Salary.info()


#Segregate the data into input and output columns
#Here salary is the target variable
ip_columns=colnames[1:13]
op_column=colnames[13]


#split the data into train and test
train_sal=Salary.iloc[:30161]
test_sal=Salary.iloc[30161:]

train_sal.shape
test_sal.shape


Xtrain=train_sal.drop(["Salary"],axis=1)
ytrain=train_sal["Salary"]

Xtest=test_sal.drop(["Salary"],axis=1)
ytest=test_sal["Salary"]

Xtrain.shape
Xtest.shape
ytrain.shape
ytest.shape


#create GaussianNB and MultinobialNB functions
sgnb=GaussianNB()

#building/fitting
Sal_fit=sgnb.fit(Xtrain,ytrain)
# predicting 
predict_model1=Sal_fit.predict(Xtest)
accuracy=round(sgnb.score(Xtrain,ytrain)*100,2)
accuracy    #79.53



smnb=MultinomialNB()
mnb_fit=smnb.fit(Xtrain,ytrain)
predict_model2=mnb_fit.predict(Xtest)
accuracy=round(smnb.score(Xtrain,ytrain)*100,2)
accuracy    #77.29


#confusionmatrix
confusion_matrix(ytest,predict_model1)
print("Accuracy",(10759+1209)/(10759+601+2491+1209))        #0.7946879150066402

confusion_matrix(ytest,predict_model2)
print("Accuracy",(10891+780)/(10891+469+2920+780))       #0.774966




