
# TASK 1: EXPLORE THE DATASET

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = "https://raw.githubusercontent.com/ArchanaInsights/Datasets/main/mobile_price.csv"
df = pd.read_csv(url)

# 1. Display first 5 rows
print("\n--- First 5 Rows ---")
print(df.head())

# 2. Dataset shape
print("\n--- Dataset Shape ---")
print(df.shape)

# 3. Column names
print("\n--- Column Names ---")
print(df.columns.tolist())

# 4. Data types
print("\n--- Data Types ---")
print(df.dtypes)

# 5. Check missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Total Missing Values ---")
print(df.isnull().sum().sum())

# 6. Statistical summary
print("\n--- Statistical Summary ---")
print(df.describe())

# 7. Correlation with Price
print("\n--- Correlation with Price ---")
correlation = df.corr(numeric_only=True)["Price"].sort_values(
    ascending=False
)
print(correlation)

# 8. Correlation heatmap focusing on Price
plt.figure(figsize=(8, 10))

price_corr = correlation.drop("Price").sort_values()

sns.barplot(x=price_corr.values, y=price_corr.index)

plt.title("Correlation of Features with Mobile Price")
plt.xlabel("Correlation with Price")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# 9. Plot top 4 correlated features against Price
top_features = (
    correlation.drop("Price")
    .abs()
    .sort_values(ascending=False)
    .head(4)
    .index
)

for feature in top_features:
    plt.figure(figsize=(6, 4))
    sns.regplot(data=df, x=feature, y="Price")
    plt.title(f"{feature} vs Price")
    plt.tight_layout()
    plt.show()

print("\n--- Task 1 Completed ---")


# TASK 2: FEATURE SELECTION AND TRAIN-TEST SPLIT
print("TASK 2 STARTED")
from sklearn.model_selection import train_test_split

# 1. Select input features
X = df[["ram", "ppi", "internal mem", "RearCam"]]

# 2. Select target variable
y = df["Price"]

print("\n--- Feature Selection ---")
print("Selected Features:", X.columns.tolist())
print("Target Variable: Price")

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 4. Display split results
print("\n--- Dataset Split Results ---")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("\n--- Task 2 Completed Successfully ---")


# TASK 3: BUILD AND TRAIN THE MODEL
print("TASK 3")
# 3.1 Create a Linear Regression Model
from sklearn.linear_model import LinearRegression

model = LinearRegression()

print("\n--- Linear Regression Model Created ---")

# 3.2 Train the Model
model.fit(X_train, y_train)

print("--- Model Training Completed Successfully ---")

print("TASK 4")

# TASK 4: EVALUATE THE MODEL

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load the dataset
url = "https://raw.githubusercontent.com/ArchanaInsights/Datasets/main/mobile_price.csv"
df = pd.read_csv(url)

# Select features and target
X = df[["ram", "ppi", "internal mem", "RearCam"]]
y = df["Price"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Build and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 4.1 Predict prices using test data
y_pred = model.predict(X_test)

print("\n--- TASK 4: MODEL EVALUATION ---")

print("\n1. Predicted vs Actual Prices")
for actual, predicted in zip(y_test.iloc[:10], y_pred[:10]):
    print(f"Actual Price: {actual:.2f} | Predicted Price: {predicted:.2f}")

# 4.2 Print coefficients and intercept
print("\n2. Coefficients (Slopes)")
for feature, coefficient in zip(X.columns, model.coef_):
    print(f"{feature}: {coefficient:.4f}")

print("\nIntercept:", round(model.intercept_, 4))

# Calculate performance metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\n3. Model Performance Metrics")
print("R2 Score:", round(r2, 4))
print("Mean Absolute Error (MAE):", round(mae, 4))
print("Mean Squared Error (MSE):", round(mse, 4))

print("\n--- TASK 4 COMPLETED ---")


print("TASK 5")

# TASK 5: CONCLUSION AND INSIGHTS

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load the dataset
url = "https://raw.githubusercontent.com/ArchanaInsights/Datasets/main/mobile_price.csv"
df = pd.read_csv(url)

# Select features and target
X = df[["ram", "ppi", "internal mem", "RearCam"]]
y = df["Price"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test prices
y_pred = model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\n--- TASK 5: CONCLUSION AND INSIGHTS ---")

# 5.1 Model Evaluation
print("\n1. Model Evaluation")

print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")

if r2 >= 0.70:
    print("The model explains a relatively high proportion of price variation.")
elif r2 >= 0.40:
    print("The model explains a moderate proportion of price variation.")
else:
    print("The model has limited predictive performance and may need improvement.")

print("A lower MAE indicates smaller average absolute prediction errors.")
print("A lower MSE indicates smaller average squared prediction errors.")

# 5.2 Insights and Discussion
print("\n2. Insights and Discussion")

# Correlation analysis
correlations = df.corr(numeric_only=True)["Price"].drop("Price")
correlations = correlations.sort_values(
    key=lambda values: values.abs(), ascending=False
)

print("\nA. Correlation Insights")
print("Features most strongly correlated with Price:")
print(correlations.head(5))

print("\nB. Selected Features")
print("The model uses RAM, PPI, internal memory, and RearCam.")
print("These features help estimate mobile prices based on their relationship with Price.")

print("\nC. Coefficients and Intercept")
for feature, coefficient in zip(X.columns, model.coef_):
    print(f"{feature} coefficient: {coefficient:.4f}")

print(f"Intercept: {model.intercept_:.4f}")
print("Each coefficient estimates the change in predicted Price for a one-unit")
print("increase in that feature, while the other selected features remain constant.")

print("\nD. Prediction Performance")
print("Compare actual and predicted prices to identify prediction differences.")
print("R2 measures explained variation, while MAE and MSE measure prediction errors.")

print("\nE. Possible Improvements")
print("1. Test additional relevant features.")
print("2. Check for outliers and unusual values.")
print("3. Compare Linear Regression with other regression models.")
print("4. Use cross-validation to assess model stability.")
print("5. Compare predictions against a baseline model.")

print("\n--- TASK 5 COMPLETED ---")
