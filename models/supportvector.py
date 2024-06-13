import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import csv
import os

# Load the dataset
data = pd.read_csv('the_data.csv')

# Define the feature columns
all_columns = ['COM-1', 'COM-2', 'POP-1', 'STA-1', 'STA-2', 'STA-3', 'STA-4', 'STA-6', 'STA-7', 'STA-8', 'STA-9', 'TEC-1', 'TEC-2', 'TEC-3', 'TEC-4']

# Split the data into features (X) and target (y)
X_features = data[all_columns]
y_target = data['status']

# Initialize the Support Vector Machine model  and StratifiedKFold cross-validation
model = SVC(probability=True)
kf = StratifiedKFold(n_splits=10)

# Initialize lists to store scores
f1_scores = []
weighted_f1_scores = []
roc_scores = []
confusion_matrices = []

# Perform StratifiedKFold cross-validation
for train_index, test_index in kf.split(X_features, y_target):
    X_train, X_test = X_features.iloc[train_index], X_features.iloc[test_index]
    y_train, y_test = y_target.iloc[train_index], y_target.iloc[test_index]

    # Train the Support Vector Machine model
    model.fit(X_train, y_train)

    # Predict and calculate scores
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate and store F1 scores
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
    f1_score = report['macro avg']['f1-score']
    weighted_f1_score = report['weighted avg']['f1-score']
    f1_scores.append(f1_score)
    weighted_f1_scores.append(weighted_f1_score)

    # Calculate and store ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    roc_scores.append(roc_auc)

    # Store the confusion matrix
    confusion_matrices.append(confusion_matrix(y_test, y_pred))


# Sum confusion matrices
final_confusion_matrix = np.sum(confusion_matrices, axis=0)

# Calculate average scores
avg_f1_score = np.mean(f1_scores)
avg_weighted_f1_score = np.mean(weighted_f1_scores)
avg_roc_score = np.mean(roc_scores)

# Initialize dictionary for storage of results
result_data = {
    'program_name': 'support_vector_machine',
    'avg_f1_score': avg_f1_score,
    'avg_weighted_f1_score': avg_weighted_f1_score,
    'avg_roc_score': avg_roc_score
}

# Define the file path for model results
output_file = 'model_results.csv'

# Write results to the file
file_exists = os.path.isfile(output_file)
with open(output_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        # Write the header if the file does not exist
        writer.writerow(['program_name', 'avg_f1_score', 'avg_weighted_f1_score', 'avg_roc_score'])
    
    # Write the results
    writer.writerow([result_data['program_name'], result_data['avg_f1_score'], result_data['avg_weighted_f1_score'], result_data['avg_roc_score']])

# Define the file path for the confusion matrix
report_file = 'confusion_matrices.txt'

# Save final confusion matrix to a text file
with open(report_file, 'a') as file:
    # Write the model name
    file.write("\nSupport Vector Machine\n")
    
    # Write the final confusion matrix
    file.write("Final Confusion Matrix:\n")
    file.write(np.array2string(final_confusion_matrix) + "\n")
