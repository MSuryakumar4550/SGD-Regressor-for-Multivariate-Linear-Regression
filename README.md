# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

Developed by : M.Suryakumar 212224040340

## Algorithm
1. Data Preparation: Select input features and target variables, split data into training and testing sets, and standardize both using scalers.  
2. Model Setup: Initialize **SGDRegressor** and wrap it with **MultiOutputRegressor** to handle multiple outputs.  
3. Model Training and Prediction: Train the model on the training set and generate predictions for the test set.  
4. Evaluation: Reverse the scaling of predictions, then calculate the **Mean Squared Error (MSE)** to measure accuracy.  


## Program:
<pre>
  import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the California Housing dataset
dataset = fetch_california_housing()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['HousingPrice'] = dataset.target
print(df.head())
  # Use the first 3 features as inputs
X = df.drop(columns=['AveOccup', 'HousingPrice'])  # Features: 'MedInc', 'HouseAge', 'AveRooms'
# Use 'MedHouseVal' and 'AveOccup' as output variables
Y = df[['AveOccup', 'HousingPrice']]  # Targets: 'MedHouseVal', 'AveOccup'

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale the features and target variables
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)
  # Initialize the SGDRegressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3)

# Use MultiOutputRegressor to handle multiple output variables
multi_output_sgd = MultiOutputRegressor(sgd)

# Train the model
multi_output_sgd.fit(X_train, Y_train)

# Predict on the test data
Y_pred = multi_output_sgd.predict(X_test)

# Inverse transform the predictions to get them back to the original scale
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

# Optionally, print some predictions
print("\nPredictions:\n", Y_pred[:5])  # Print first 5 predictions
</pre>

## Output:
![multivariate linear regression model for predicting the price of the house and number of occupants in the house](sam.png)
![image](https://github.com/user-attachments/assets/c3f9876d-9777-4edf-b103-45fb68015c45)
![image](https://github.com/user-attachments/assets/f9bd71b3-a439-4d08-b040-0a4889fa1a15)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
