# Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Loading file
file_path = '/Users/meliscan/machineProject/electricity_data.csv'

data = pd.read_csv(file_path)

# Combining the Date and Time columns and converting them to datetime type
data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')

# Filtering rows where the minute is 0
hourly_data = data[data['datetime'].dt.minute == 0]

# Removing the original Date and Time columns
del hourly_data['Date']
del hourly_data['Time']

# Saving or inspecting the filtered dataset
hourly_data.reset_index(inplace=True, drop=True)  # To reset the index
print(hourly_data.head())

file_path2 = '/Users/meliscan/machineProject/hourly_electricity_data.csv' 
hourly_data = pd.read_csv(file_path2)

hourly_data['datetime'] = pd.to_datetime(hourly_data['datetime'])

# Extract year, month, day, hour by adding new columns
hourly_data['year'] = hourly_data['datetime'].dt.year
hourly_data['month'] = hourly_data['datetime'].dt.month
hourly_data['day'] = hourly_data['datetime'].dt.day
hourly_data['hour'] = hourly_data['datetime'].dt.hour
hourly_data['weekday'] = hourly_data['datetime'].dt.weekday  # day of the week (0: monday, 6: sunday)

hourly_data = hourly_data.drop(columns=['datetime'])

# Remove missing data
hourly_data = hourly_data.dropna()

print(hourly_data.head())

# Statistical summary
print(hourly_data.describe())

# Separate input (X) and target (y) variables
X = hourly_data.drop(columns=['Global_active_power']) 
y = hourly_data['Global_active_power'] 

# Split into training and temporary set (test + validation)
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Split the test and validation sets
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Training Data Size:", X_train.shape)
print("Test Data Size:", X_test.shape)
print("Validation Data Size:", X_val.shape)

# Building the model
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Input layer specifying the input shape
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),  # First layer with L2 regularization 
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1, activation='linear')  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Display the model summary
model.summary()

# Training the model
history = model.fit(
    X_train, y_train,
    epochs=25, 
    batch_size=32, 
    validation_data=(X_val, y_val), 
    verbose=1  
) 

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.title('Model MAE Over Epochs')
plt.legend()
plt.show()

# Model predictions
y_pred = model.predict(X_val)

# Real vs Prediction Graph
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--') 
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()

# Residuals Analysis
residuals = y_val - y_pred.flatten()

plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='purple', bins=30)
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Define daytime (09:00 - 18:00) and nighttime (18:00 - 09:00) hours
hourly_data['period'] = hourly_data['hour'].apply(lambda x: 'Day' if 9 <= x <= 18 else 'Night')

# Calculate hourly average consumption for daytime and nighttime
day_consumption = hourly_data[hourly_data['period'] == 'Day'].groupby('hour')['Global_active_power'].mean()
night_consumption = hourly_data[hourly_data['period'] == 'Night'].groupby('hour')['Global_active_power'].mean()

# Calculate the 75th percentile thresholds for daytime and nighttime
day_threshold = day_consumption.quantile(0.75)
night_threshold = night_consumption.quantile(0.75)

# Identify the hours where consumption exceeds the threshold for daytime and nighttime
day_high_usage = day_consumption[day_consumption > day_threshold]
night_high_usage = night_consumption[night_consumption > night_threshold]

# Warning messages
print("=== Daytime (09:00 - 18:00) ===")
if not day_high_usage.empty:
    print("Warning! High consumption hours during daytime:")
    for hour, consumption in day_high_usage.items():
        print(f"  - Hour {hour}: {consumption:.2f} kW (exceeds threshold!)")
else:
    print("Congratulations! No hours exceed the threshold during daytime.")

print("\n=== Nighttime (18:00 - 09:00) ===")
if not night_high_usage.empty:
    print("Warning! High consumption hours during nighttime:")
    for hour, consumption in night_high_usage.items():
        print(f"  - Hour {hour}: {consumption:.2f} kW (exceeds threshold!)")
else:
    print("Congratulations! No hours exceed the threshold during nighttime.")

    # Plotting the daytime and nighttime graphs
plt.figure(figsize=(12, 6))

# Daytime plot
plt.subplot(1, 2, 1)
plt.plot(day_consumption.index, day_consumption.values, marker='o', label='Day Consumption')
plt.axhline(y=day_threshold, color='red', linestyle='--', label=f'Threshold ({day_threshold:.2f} kW)')
plt.scatter(day_high_usage.index, day_high_usage.values, color='orange', label='High Consumption Hours')
plt.title('Day Consumption and Threshold')
plt.xlabel('Hour')
plt.ylabel('Average Consumption (kW)')
plt.legend()
plt.grid()

# Nighttime plot
plt.subplot(1, 2, 2)
plt.plot(night_consumption.index, night_consumption.values, marker='o', label='Night Consumption')
plt.axhline(y=night_threshold, color='red', linestyle='--', label=f'Threshold ({night_threshold:.2f} kW)')
plt.scatter(night_high_usage.index, night_high_usage.values, color='orange', label='High Consumption Hours')
plt.title('Night Consumption and Threshold')
plt.xlabel('Hour')
plt.ylabel('Average Consumption (kW)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Calculate the average consumption for each month
monthly_consumption = hourly_data.groupby('month')['Global_active_power'].mean()

# Threshold for monthly consumption (75th percentile)
monthly_threshold = monthly_consumption.quantile(0.75)

# Identify the months where consumption exceeds the threshold
high_usage_months = monthly_consumption[monthly_consumption > monthly_threshold]

# Warning messages
print("=== Monthly Analysis ===")
if not high_usage_months.empty:
    print("Warning! Months exceeding the threshold:")
    for month, consumption in high_usage_months.items():
        print(f"  - Month {month}: {consumption:.2f} kW (exceeds threshold!)")
else:
    print("Congratulations! No months exceed the threshold.")

    # Summer and Winter months analysis
summer_months = hourly_data[hourly_data['month'].isin([6, 7, 8])]
winter_months = hourly_data[hourly_data['month'].isin([12, 1, 2])]

summer_consumption = summer_months['Global_active_power'].mean()
winter_consumption = winter_months['Global_active_power'].mean()

print("Average Consumption in Summer Months:", summer_consumption)
print("Average Consumption in Winter Months:", winter_consumption)

# Create a plot
plt.figure(figsize=(10, 6))
plt.bar(monthly_consumption.index, monthly_consumption.values, color='skyblue', label='Monthly Average Consumption')
plt.axhline(y=monthly_threshold, color='red', linestyle='--', label=f'Threshold ({monthly_threshold:.2f} kW)')
plt.bar(high_usage_months.index, high_usage_months.values, color='orange', label='High Consumption Months')
plt.xticks(monthly_consumption.index, ['January', 'February', 'March', 'April', 'May', 'June', 
                                       'July', 'August', 'September', 'October', 'November', 'December'], rotation=45)
plt.title('Monthly Average Consumption and Threshold')
plt.xlabel('Month')
plt.ylabel('Average Consumption (kW)')
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Daily total consumption
weekday_consumption = hourly_data.groupby('weekday')['Global_active_power'].mean()

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plt.figure(figsize=(10, 6))
plt.bar(days, weekday_consumption.values, color='green')
plt.title('Average Consumption by Day of the Week')
plt.xlabel('Day')
plt.ylabel('Average Consumption (kW)')
plt.show()


