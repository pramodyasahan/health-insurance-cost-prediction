# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Loading the dataset from a CSV file
dataset = pd.read_csv('insurance.csv')

# Extracting features (X) and target variable (y) from the dataset
X = dataset.iloc[:, :-1].values  # All rows, all columns except the last one
y = dataset.iloc[:, -1].values  # All rows, only the last column

# Applying OneHotEncoder to categorical columns (sex, smoker, region)
# This transforms categorical data into binary (0/1) values for each category
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 4, 5])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Scaling the features (Standardization)
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Scaling the target variable
sc_y = StandardScaler()
y = y.reshape(-1, 1)  # Reshaping y to a 2D array for the scaler
y = sc_y.fit_transform(y)
y = y.ravel()  # Flattening y back to a 1D array

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Creating polynomial features (degree 4 and only interaction terms)
poly_reg = PolynomialFeatures(degree=4, interaction_only=True)
X_poly_train = poly_reg.fit_transform(X_train)  # Transforming training set
X_poly_test = poly_reg.transform(X_test)  # Transforming test set

# Creating and training the Linear Regression model with Polynomial features
lin_reg = LinearRegression()
lin_reg.fit(X_poly_train, y_train)

# Predicting the target variable for the test set
y_pred = lin_reg.predict(X_poly_test)

# Setting display options for numpy arrays
np.set_printoptions(precision=2)

# Printing the predicted and actual values side by side for comparison
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Creating an index range for plotting
index = range(len(y_pred))

# Plotting Predicted vs Actual values
plt.plot(index, y_pred, color='red', label='Predicted Grades')
plt.plot(index, y_test, color='blue', label='Actual Grades')

# Adding title, labels, and legend to the plot
plt.title('Comparison of Predicted and Actual Grades')
plt.xlabel('Index of Samples')
plt.ylabel('Grades')
plt.legend()

# Displaying the plot
plt.show()
