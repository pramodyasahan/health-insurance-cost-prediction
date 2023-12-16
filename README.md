
---

# Health Insurance Cost Prediction using Polynomial Regression

## Overview
This project focuses on predicting health insurance costs using a polynomial regression model. By employing machine learning techniques in Python, the project aims to accurately estimate insurance costs based on various personal attributes. The model takes into account several features including age, sex, BMI, number of children, smoking status, and region to predict individual medical costs billed by health insurance.

## Dataset
The dataset used in this project is sourced from a CSV file named `insurance.csv`. It includes several factors that are typically influential in determining health insurance costs:
- Age
- Sex
- BMI (Body Mass Index)
- Number of Children
- Smoking Status
- Region

The target variable is the individual medical costs billed by health insurance.

## Methodology

### Data Preprocessing
- **Categorical Encoding**: Categorical features (Sex, Smoker, Region) are transformed into binary (0/1) values using OneHotEncoder, facilitating their use in the model.
- **Feature Scaling**: StandardScaler is applied to standardize the features, ensuring that all features contribute equally to the model's prediction.

### Model Training
- **Polynomial Feature Transformation**: PolynomialFeatures is used to generate polynomial and interaction terms up to the fourth degree, capturing more complex relationships in the data.
- **Linear Regression**: A Linear Regression model is then trained on these polynomial features.

### Splitting the Dataset
The dataset is split into training and testing sets, with 80% of the data used for training and 20% for testing. This split allows for effective model evaluation.

### Model Evaluation
- **Prediction**: The model predicts health insurance costs on the test set.
- **Comparison**: Predicted values are compared against actual values to evaluate the model's performance.

### Visualization
- **Plotting**: The project includes a plot comparing predicted and actual values, providing a visual assessment of the model's performance.

## Execution
The project is executed in Python, utilizing libraries such as Pandas for data manipulation, NumPy for numerical operations, Matplotlib for plotting, and Scikit-learn for machine learning models.

## Running the Project
To run the project:
1. Ensure you have Python and the necessary libraries installed.
2. Clone the repository and navigate to the project directory.
3. Run the Python script to train the model and visualize the results.

## Conclusion
This project demonstrates the application of polynomial regression in predicting health insurance costs. The use of polynomial features allows the model to capture more complex relationships than standard linear regression, potentially leading to more accurate predictions.

---
