############### Decision Tree ######################

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Save the model
joblib.dump(classifier, '/content/drive/MyDrive/Dataset2/models/decision_tree_model.pkl')

# Make predictions
y_train_pred = classifier.predict(X_train)
y_test_pred = classifier.predict(X_test)

# Calculate evaluation metrics
metrics_train = {
    "Accuracy": accuracy_score(y_train, y_train_pred),
    "Recall": recall_score(y_train, y_train_pred, average='weighted'),
    "Precision": precision_score(y_train, y_train_pred, average='weighted'),
    "F1 Score": f1_score(y_train, y_train_pred, average='weighted')
}
metrics_test = {
    "Accuracy": accuracy_score(y_test, y_test_pred),
    "Recall": recall_score(y_test, y_test_pred, average='weighted'),
    "Precision": precision_score(y_test, y_test_pred, average='weighted'),
    "F1 Score": f1_score(y_test, y_test_pred, average='weighted')
}

# Print the metrics
print("Training Metrics (Accuracy, Recall, Precision, F1):", metrics_train)
print("Testing Metrics (Accuracy, Recall, Precision, F1):", metrics_test)

# Plot the metrics
metrics_df = pd.DataFrame([metrics_train, metrics_test], index=['Train', 'Test'])

plt.figure(figsize=(10, 6))
ax = metrics_df.plot(kind='bar')
plt.title('Evaluation Metrics for Decision Tree Classifier')
plt.xlabel('Data Split')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.grid(axis='y')

# Update legend with exact metric values
handles, labels = ax.get_legend_handles_labels()
new_labels = [f'{label} - Train: {metrics_train[label]:.2f}, Test: {metrics_test[label]:.2f}' for label in labels]
ax.legend(handles, new_labels, loc='lower right')

# Save the metrics plot
metrics_plot_path = '/content/drive/MyDrive/Dataset2/models/decision_tree_metrics.png'
plt.savefig(metrics_plot_path)
print(f"Metrics plot saved to {metrics_plot_path}")

# Plot confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=ax[0], cmap='Blues', colorbar=False)
ax[0].set_title('Confusion Matrix - Train Set')

ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, ax=ax[1], cmap='Blues', colorbar=False)
ax[1].set_title('Confusion Matrix - Test Set')

plt.tight_layout()

# Save the confusion matrices plot
conf_matrix_plot_path = '/content/drive/MyDrive/Dataset2/models/decision_tree_confusion_matrices.png'
plt.savefig(conf_matrix_plot_path)
print(f"Confusion matrices plot saved to {conf_matrix_plot_path}")

plt.show()
