import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle

# Load and clean the data
df = pd.read_csv(r'C:/Users/shajea/Downloads/Solar_Prediction.csv/Solar_Prediction.csv')
df = df.dropna()  # Remove rows with missing values
df = df[df['Radiation'] >= 0]  # Remove negative values for solar radiation

# Convert time columns to datetime and extract useful features
df['TimeSunRise'] = pd.to_datetime(df['TimeSunRise'], format='%H:%M:%S').dt.hour + pd.to_datetime(df['TimeSunRise'], format='%H:%M:%S').dt.minute / 60
df['TimeSunSet'] = pd.to_datetime(df['TimeSunSet'], format='%H:%M:%S').dt.hour + pd.to_datetime(df['TimeSunSet'], format='%H:%M:%S').dt.minute / 60

# Define features and target variable
X = df[['Temperature', 'Humidity', 'Pressure', 'Speed', 'TimeSunRise', 'TimeSunSet']]
y = df['Radiation']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model and pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('scaler', StandardScaler()),  # Standardize the features
    ('regressor', GradientBoostingRegressor(random_state=42))  # Model
])

# Perform Grid Search for hyperparameter tuning
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__learning_rate': [0.01, 0.05, 0.1]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Save the best model
model_path = r'C:/Users/shajea/Desktop/solar_model.pkl'
with open(model_path, 'wb') as model_file:
    pickle.dump(grid_search.best_estimator_, model_file)

# Print performance metrics
y_pred = grid_search.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'R^2 Score: {r2:.2f}')
print(f'Mean Absolute Error: {mae:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')

# Cross-validation
cv_scores = cross_val_score(grid_search.best_estimator_, X, y, cv=5, scoring='r2')
print(f'Cross-Validation R^2 Scores: {cv_scores}')
print(f'Average Cross-Validation R^2 Score: {cv_scores.mean():.2f}')
