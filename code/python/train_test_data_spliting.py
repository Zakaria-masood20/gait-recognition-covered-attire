import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset with the specified encoding
data = pd.read_csv('/content/drive/MyDrive/Dataset2/frames_00_processed/gait_data_sorted.csv')

# Assuming the last column is the target
feature_columns = data.columns[:-1]  # All columns except the last one
target_column = data.columns[-1]     # Last column as target

# Prepare features (X) and targets (y)
X = data[feature_columns]
y = data[target_column]

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Optionally, verify the stratification and the split
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Save the splits to new CSV files for future use
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv('/content/drive/MyDrive/Dataset2/train_test/gait_data.train.csv', index=False)
test_data.to_csv('/content/drive/MyDrive/Dataset2/train_test/gait_data.test.csv', index=False)

print("Data has been split and saved into training and testing CSV files.")
