import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
# To remove the scientific notation from numpy arrays
np.set_printoptions(suppress=True)

def preparationDatasetTaskFingerprinting():
    """Prepare the data from the csv file which was obtained from the task fingeprinting technique, to subsequently be used for the development of a deep learning model."""
    #TODO the inputs will be based on the csv files: one for the target variables and another one for the predictors
    # from the user input, it must be indicated the file with the task fingerprinting data, to be then processed in this script the deep learning model
    data=pd.read_csv("", sep=';')
    data.head()

    #convert all NaN to -1
    data = data.fillna(-1)
    data.head()

    #from the user input, it will be assigned the target variables (usually related with the accuracy metrics to predict) and the predictors (usually related with the independent variables)
    TargetVariable=[]
    Predictors=[]

    X=data[Predictors].values
    y=data[TargetVariable].values

    ### Standardization of data ###

    PredictorScaler=StandardScaler()
    TargetVarScaler=StandardScaler()

    # Storing the fit object for later reference
    PredictorScalerFit=PredictorScaler.fit(X)
    TargetVarScalerFit=TargetVarScaler.fit(y)

    # Generating the standardized values of X and y
    X=PredictorScalerFit.transform(X)
    y=TargetVarScalerFit.transform(y)
    return {"stdValuesX":X, "stdValuesY": y}

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#normalize all sets
from sklearn.preprocessing import Normalizer
norm = Normalizer()
X_train = norm.fit_transform(X_train)
X_test = norm.fit_transform(X_test)
y_train = norm.fit_transform(y_train)
y_test = norm.fit_transform(y_test)
"""
#print normalized data
print(X_train)
print(y_train)
print(X_test)
print(y_test)
"""

# Quick sanity check with the shapes of Training and testing datasets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# importing the libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

# create ANN model
model = Sequential()

# Defining the Input layer and FIRST hidden layer, both are same!
model.add(Dense(units=256, input_dim=105, kernel_initializer='normal', activation='relu'))

# Defining the Second layer of the model
# after the first layer we don't have to specify input_dim as keras configure it automatically
model.add(Dropout(0.15))
model.add(Dense(units=128, kernel_initializer='normal', activation='tanh'))
model.add(Dropout(0.15))
model.add(Dense(units=64, kernel_initializer='normal', activation='tanh'))
model.add(Dropout(0.15))
model.add(Dense(units=32, kernel_initializer='normal', activation='tanh'))
model.add(Dropout(0.15))
model.add(Dense(units=16, kernel_initializer='normal', activation='tanh'))
model.add(Dropout(0.15))
model.add(Dense(units=8, kernel_initializer='normal', activation='tanh'))

# The output neuron is a single fully connected node
# Since we will be predicting a single number
model.add(Dense(4, kernel_initializer='normal'))

# Compiling the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Fitting the ANN to the Training set
model.fit(X_train, y_train ,batch_size = 20, epochs = 700, verbose=1)

model.summary()
model.save('modelo.h5')

from keras_sequential_ascii import keras2ascii
keras2ascii(model)
import tensorflow as tf

tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
    layer_range=None,
    show_layer_activations=True,
)

"""# Accuracy"""

# Fitting the ANN to the Training set
model.fit(X_train, y_train ,batch_size = 15, epochs = 5, verbose=0)

# Generating Predictions on testing data
Predictions=model.predict(X_test)

# Scaling the predicted Price data back to original price scale
Predictions=TargetVarScalerFit.inverse_transform(Predictions)

# Scaling the y_test Price data back to original price scale
y_test_orig=TargetVarScalerFit.inverse_transform(y_test)

# Scaling the test data back to original scale
Test_Data=PredictorScalerFit.inverse_transform(X_test)

#print(Predictions)
#print(y_test_orig)
#['classificationAccuracy','countingAccuracy','sentimentAnalysisAccuracy','transcriptionAccuracy']

TestingData=pd.DataFrame(data=Test_Data, columns=Predictors)

#based on the user input for the variables to predict the accuracy
TestingData['']=y_test_orig[:,[0]]
TestingData['Predicted']=Predictions[:,[0]]
TestingData.head()

# Computing the absolute percent error
APE=100*(abs(TestingData['classificationAccuracy']-TestingData['PredictedClassificationAccuracy'])/TestingData['classificationAccuracy'])
TestingData['APE']=APE

print('The Accuracy of ANN model is:', 100-np.mean(APE))
TestingData.head()

# Computing the absolute percent error
APE=100*(abs(TestingData['classificationAccuracy']-TestingData['PredictedClassificationAccuracy'])/TestingData['classificationAccuracy'])
TestingData['APE']=APE

print('The Accuracy of ANN model is:', 100-np.mean(APE))
TestingData.head()

# ask user data to be predicted, split by ; and convert into numpy array
print("Enter the data to be predicted:")
data_to_predict = input()
data_to_predict = np.array(data_to_predict.split(','))
data_to_predict = data_to_predict.astype(float)
data_to_predict = data_to_predict.reshape(1, -1)
data_to_predict = PredictorScalerFit.transform(data_to_predict)
data_to_predict = norm.fit_transform(data_to_predict)

y_pred = model.predict(data_to_predict)
y_pred = TargetVarScaler.inverse_transform(y_pred)
print(y_pred)