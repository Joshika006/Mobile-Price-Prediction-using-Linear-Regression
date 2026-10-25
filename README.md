# Mobile Price Prediction using Linear Regression

## Project Overview
This project predicts mobile phone prices using a Linear Regression machine learning model. The dataset is explored, relevant features are selected, and the model is trained and evaluated using test data.

## Objectives
- Explore the mobile price dataset.
- Analyze feature correlations with mobile prices.
- Select relevant features for prediction.
- Build and train a Linear Regression model.
- Evaluate model performance using regression metrics.
- Discuss insights and possible improvements.

## Dataset
Dataset: [Mobile Price Dataset](https://raw.githubusercontent.com/ArchanaInsights/Datasets/main/mobile_price.csv)

The dataset contains mobile phone features such as RAM, PPI, internal memory, rear camera, battery, thickness, and Price.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Features Selected
The following features are used to predict mobile prices:
- RAM
- PPI
- Internal Memory
- Rear Camera

**Target Variable:** Price

## Project Workflow
1. **Data Exploration:** Inspect dataset shape, columns, data types, missing values, statistical summaries, and correlations.
2. **Feature Selection:** Select relevant input features and split the data into training and testing sets.
3. **Model Building:** Create and train a Linear Regression model.
4. **Model Evaluation:** Generate predictions and calculate the coefficients, intercept, R² Score, Mean Absolute Error (MAE), and Mean Squared Error (MSE).
5. **Conclusion:** Interpret the results, discuss insights, and identify possible improvements.

## Model Evaluation Metrics
- **R² Score:** Measures how well the model explains variation in mobile prices.
- **MAE:** Measures the average absolute prediction error.
- **MSE:** Measures the average squared prediction error.

Actual metric values should be added after running the model.

## How to Run
1. Install Python.
2. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. Run the Python file:

   ```bash
   python Mobile_Price_Prediction.py
   ```

An internet connection is required to load the dataset from the provided URL.

## Conclusion
This project demonstrates how Linear Regression can be used to predict mobile phone prices using selected mobile specifications. The model's effectiveness is assessed using regression evaluation metrics. Further improvements may include testing additional features, comparing other regression algorithms, and using cross-validation.

## Author
Amala Joshika A
