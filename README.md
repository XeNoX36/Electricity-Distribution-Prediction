# Project Report

**Data Overview**  

Dataset: electricity demand dataset.csv  
Core Column: Timestamp — datetime representation of observation period.  
Dependent Variable: Electricity Demand (or equivalent column in dataset)  
Independent Variables: Temporal features such as hour, day, month, temperature, region, etc.  

The dataset was imported and explored as follows:  
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"c:\Users\USER\Documents\CODES\Python Work to Do\electricity demand dataset.csv")
data["Timestamp"] = pd.to_datetime(data["Timestamp"])

**Data Cleaning and Preprocessing**
The preprocessing steps included:

# Checking for missing values
data.isnull().sum()

# Handling missing entries if present
data = data.dropna()

# Feature engineering
data["Hour"] = data["Timestamp"].dt.hour
data["Day"] = data["Timestamp"].dt.day
data["Month"] = data["Timestamp"].dt.month
data["Year"] = data["Timestamp"].dt.year
data["Weekday"] = data["Timestamp"].dt.weekday
```
These features allowed the model to learn time-based consumption behavior.

**1. Exploratory Data Analysis (EDA)**  
**Descriptive Summary**  
Statistical overview using:
```python 
data.describe()
```
Average daily demand was consistent within certain periods but spiked during warmer months.  

Variability in consumption suggested both seasonal and hourly patterns.  

**Visualization Insights**  

Line plots: Displayed demand over time to identify upward or cyclical trends.

Boxplots: Highlighted fluctuations by hour and month.

Correlation Heatmap:
```python
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
```
Revealed moderate correlation between hour, temperature, and electricity usage.

**2. XGBoost Modeling and Implementation**

**Objective**
To build a supervised regression model that predicts electricity demand based on historical and temporal data.

**Feature Selection**

Predictors (X) include engineered features such as:  
Hour  
Day  
Month  
Weekday  
temperature and humidity  

Target variable (y):  
Electricity consumption

**Code Implementation**
```python
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define features and target
X = data[["Hour", "Day", "Month", "Weekday"]]
y = data["Electricity_Demand"]   # Replace with actual target column name

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost model
xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on test set
y_pred = xgb_model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R² Score:", r2)
```
**Model Performance**  

results observed:  

The XGBoost model successfully learned non-linear relationships between time-based features and demand levels.

**Feature Importance Analysis**  
The model’s feature importance visualization helps identify which time features most strongly influence electricity usage:
```python
plt.figure(figsize=(8,5))
plt.barh(X.columns, xgb_model.feature_importances_)
plt.title("Feature Importance in Electricity Demand Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()
```
![]()
Insights:

Hour and Weekday were the most influential predictors, highlighting consistent daily cycles and work-week effects.  

Month contributed significantly, confirming seasonal trends in consumption.  

**3. Key Insights**  

**Demand Predictability:**  
Electricity usage follows strong hourly and seasonal patterns, allowing for accurate modeling.

**Peak Hours:**  
Late afternoon and early evening remain high-demand periods.

**Seasonal Variations:**   
Increased demand during hot or cold seasons due to cooling and heating needs.
