#########################ANN#####################
####################forestfires.csv#################
#PREDICT THE BURNED AREA OF FOREST FIRES WITH NEURAL NETWORKS

#import libraries
import pandas as pd
import numpy as 
#load the dataset
forest_data=pd.read_csv("C:/Users/server/Downloads/forestfires (1).csv")
forest_data.head()
forest_data.shape   #(517, 31)
forest_data.dtypes

#check any null value present in the data set
forest_data.isnull().sum()  #No null values

forest_data["month"].value_counts()

forest_data["day"].value_counts()

forest_data.info()


# using groupby method to group month columns
g=forest_data.groupby('month')
for month,month_df in g:
    print(month)
    print(month_df)
    
g.get_group("aug")

# using groupby method to group size_category columns
g=forest_data.groupby('size_category')
for size,size_df in g:
    print(size)
    print(size_df)
    
g.get_group("small")

#######Visualization
import seaborn as sns
import matplotlib.pyplot as plt
#visualize number of size_category
sns.set_style("whitegrid")
sns.countplot(x="size_category",data=forest_data)

#visualize number of month category
sns.set_style("whitegrid")
sns.countplot(x="month",data=forest_data)

#using box chart to visualize temperature on month  basis using seaborn
plt.figure(figsize=(12,7))
sns.boxplot(x='month',y='temp',data=forest_data,palette='winter')
#here we can see that the temperature on month basis. here the month july has showing highest temperature, month december showing lowest temperature

forest_data.groupby('day').rain.count().plot()

sns.distplot(forest_data["temp"])

#label encoding to convert categorical data to numeric format
strings=["month","day","size_category"]
from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in strings:
     forest_data[i] = number.fit_transform(forest_data[i])
forest_data.dtypes
forest_data.head()

####Model building
###Split the data into train and test
#df.iloc[:,0:5]
X=forest_data.iloc[:,0:30]  
y = forest_data['size_category']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train.shape,y_train.shape,X_test.shape,y_test.shape  #((387, 30), (387,), (130, 30), (130,))

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

# Initializing the ANN
classifier = Sequential()

#creating input layers and first hidden layers
classifier.add(Dense(output_dim = 6, init = 'he_uniform',activation='relu',input_dim = 30))

#creating second input layer
classifier.add(Dense(output_dim = 6, init = 'he_uniform',activation='relu'))

#creating output layers
classifier.add(Dense(output_dim = 1, init = 'glorot_uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, nb_epoch = 100)

print(model_history.history.keys())  #dict_keys(['val_loss', 'val_accuracy', 'loss', 'accuracy'])

# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Make predictions using test data
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred


# Calculate the Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)   #0.7153846153846154

# create Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
#array([[ 0, 37],
 #      [ 0, 93]], dtype=int64)

#To improve model accuracy, we have to perform hyper parameter tuning
###Hyper parameter tuning
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
# Function to create model,for KerasClassifier
def create_model(optimizer='adam'):
    #defining my model
    model = Sequential()
    model.add(Dense(12, input_dim=30, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# create model
model = KerasClassifier(build_fn=create_model)

# define the grid search parameters
batchSize = [10, 20, 40, 60, 80, 100]
epochs = [10, 30, 50]
optimizer = ['SGD','Adadelta', 'RMSprop', 'Adagrad','Adam']

parameter_grid = dict(optimizer=optimizer,batch_size=batchSize, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=parameter_grid, n_jobs=-1, cv=3)
ANN_grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (ANN_grid_result.best_score_, ANN_grid_result.best_params_))
#Best: 0.762274 using {'batch_size': 10, 'epochs': 50, 'optimizer': 'Adam'}
#Model accuracy improved

#lets create final model using best parameter
# Compiling the ANN
classifier.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
final_model=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, nb_epoch = 50)

print(final_model.history.keys()) #dict_keys(['val_loss', 'val_accuracy', 'loss', 'accuracy'])


# summarize history for accuracy
plt.plot(final_model.history['accuracy'])
plt.plot(final_model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(final_model.history['loss'])
plt.plot(final_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#####prediction
pred = classifier.predict(X_test)
pred = (pred > 0.5)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(pred,y_test)  #0.7615384615384615

# create Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)
#array([[ 6, 31],
 #      [ 0, 93]], dtype=int64)


pred_data=pd.DataFrame(pred)
final_data=pd.concat([forest_data[["size_category"]],pred_data],axis=1)
final_data.columns=["actual_data","predicted_data"]
final_data.head(50)


