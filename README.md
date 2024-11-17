# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Data Collection**:  
   - Import essential libraries like pandas, numpy, sklearn, matplotlib, and seaborn.  
   - Load the dataset using `pandas.read_csv()`.

2. **Data Preprocessing**:  
   - Address any missing values in the dataset.  
   - Select key features for training the models.  
   - Split the dataset into training and testing sets with `train_test_split()`.

3. **Linear Regression**:  
   - Initialize the Linear Regression model from sklearn.  
   - Train the model on the training data using `.fit()`.  
   - Make predictions on the test data using `.predict()`.  
   - Evaluate model performance with metrics such as Mean Squared Error (MSE) and the R² score.

4. **Polynomial Regression**:  
   - Use `PolynomialFeatures` from sklearn to create polynomial features.  
   - Fit a Linear Regression model to the transformed polynomial features.  
   - Make predictions and evaluate performance similar to the linear regression model.

5. **Visualization**:  
   - Plot the regression lines for both Linear and Polynomial models.  
   - Visualize residuals to assess model performance.

## Program:
```python
'''
Program to implement Linear and Polynomial Regression models for predicting car prices and test assumptions.
Developed by: Sanjushri A
RegisterNumber: 21223040187
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv"
data = pd.read_csv(url)

# Select relevant features and target variable
X = data[['enginesize']]  # Predictor
y = data['price']         # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Linear Regression ----
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# ---- Polynomial Regression ----
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# ---- Assumptions Testing ----

# 1. Linearity: Scatterplot of actual vs predicted values
plt.scatter(y_test, y_pred_linear, color='blue')
plt.title('Linearity Check: Actual vs Predicted')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.axline([0, 0], [1, 1], color='red', linestyle='--')  # Perfect fit line
plt.show()

# 2. Homoscedasticity: Residuals vs predicted values
residuals = y_test - y_pred_linear
plt.scatter(y_pred_linear, residuals, color='blue')
plt.title('Homoscedasticity Check: Residuals vs Predicted')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.axhline(0, color='red', linestyle='--')
plt.show()

# 3. Normality: Histogram of residuals
sns.histplot(residuals, kde=True, color='blue')
plt.title('Normality Check: Residuals')
plt.xlabel('Residuals')
plt.show()

# 4. Multicollinearity: VIF (only for multiple predictors)
# Add this section if you include multiple predictors
# X_features = data[['horsepower', 'curbweight', 'enginesize', 'highwaympg']]
# vif_data = pd.DataFrame()
# vif_data['Feature'] = X_features.columns
# vif_data['VIF'] = [variance_inflation_factor(X_features.values, i) for i in range(X_features.shape[1])]
# print("VIF Data:\n", vif_data)

# ---- Evaluation Metrics ----
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_linear))
print("Linear Regression R² Score:", r2_score(y_test, y_pred_linear))
print("Polynomial Regression MSE:", mean_squared_error(y_test, y_pred_poly))
print("Polynomial Regression R² Score:", r2_score(y_test, y_pred_poly))
```

## Output:
![image](https://github.com/user-attachments/assets/497f1c8b-0efe-4c00-a286-30b24543f2f1)
![image](https://github.com/user-attachments/assets/0043cf54-9835-48b0-b654-6c1bd162d0af)
![image](https://github.com/user-attachments/assets/fe162309-423d-4fe2-a973-642dba9bd9c9)
![image](https://github.com/user-attachments/assets/9e875a9f-6120-4cbb-a9f2-245fee179ea9)



## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
