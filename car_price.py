# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars'
data = pd.read_csv(file_path)

# Preprocessing
# Calculate car's age
data['Car_Age'] = 2024 - data['Year']

# Drop irrelevant columns
data = data.drop(['Car_Name', 'Year'], axis=1)

# Encode categorical variables
label_encoder = LabelEncoder()
data['Fuel_Type'] = label_encoder.fit_transform(data['Fuel_Type'])
data['Selling_type'] = label_encoder.fit_transform(data['Selling_type'])
data['Transmission'] = label_encoder.fit_transform(data['Transmission'])

# Split the dataset into features (X) and target (y)
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

# Optional: Save the trained model for future use
# import joblib
# joblib.dump(model, 'car_price_prediction_model.pkl')
