import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('insurance.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 4, 5])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

sc_X = StandardScaler()
X = sc_X.fit_transform(X)

sc_y = StandardScaler()
y = y.reshape(-1, 1)
y = sc_y.fit_transform(y)
y = y.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

poly_reg = PolynomialFeatures(degree=4, interaction_only=True)
X_poly_train = poly_reg.fit_transform(X_train)
X_poly_test = poly_reg.transform(X_test)  # Transform the test set

lin_reg = LinearRegression()
lin_reg.fit(X_poly_train, y_train)

y_pred = lin_reg.predict(X_poly_test)

np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

index = range(len(y_pred))

plt.plot(index, y_pred, color='red', label='Predicted Grades')
plt.plot(index, y_test, color='blue', label='Actual Grades')

plt.title('Comparison of Predicted and Actual Grades')
plt.xlabel('Index of Samples')
plt.ylabel('Grades')
plt.legend()

plt.show()
