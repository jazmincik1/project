from sklearn.model_selection import GridSearchCV
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imutils import paths

def import_images_data(samples_directory):
    image_pixels = []
    image_labels = []

    for i, image_path in enumerate(samples_directory):
        sample_image = cv2.imread(image_path)
        sample_label = image_path.split(os.path.sep)[-2]
        sample_image = cv2.resize(sample_image, (32, 32), interpolation=cv2.INTER_AREA)
        image_pixels.append(sample_image)
        image_labels.append(sample_label)

    return (np.array(image_pixels), np.array(image_labels))

# Directory containing image files
#path_to_sample_images = 'raw-img1'
path_to_sample_images = '/content/drive/My Drive/colab_data/animals'

param_grid = {
    'C': [0.01,0.1,1],  # Penalty parameter C
    'kernel': ['linear']  # Kernel type
}

# List all images in the directory
samples_directory = list(paths.list_images(path_to_sample_images))
samples_directory_half = samples_directory[:len(samples_directory)]

# Import and prepare image data
(image_pixels_half, image_labels_half) = import_images_data(samples_directory_half)

# Reshape image pixels for SVM input
image_pixels_half = image_pixels_half.reshape((image_pixels_half.shape[0], 3072))

# Encode labels
encoder = LabelEncoder()
image_labels_half = encoder.fit_transform(image_labels_half)

# Split the data into train, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(image_pixels_half, image_labels_half, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 25% of 80% = 20%

print("Starting Classification...")

# Create and train the SVM classifier with GridSearchCV
svm_classifier = SVC()
grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the SVM classifier with best hyperparameters
svm_classifier_best = SVC(**best_params)
svm_classifier_best.fit(X_train, y_train)

# Predict and evaluate the model
y_pred_val = svm_classifier_best.predict(X_val)
y_pred_test = svm_classifier_best.predict(X_test)

accuracy_val = accuracy_score(y_val, y_pred_val)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f'Validation Accuracy: {accuracy_val}')
print(f'Test Accuracy: {accuracy_test}')

# Confusion matrix for the test data
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap (Test Data)')
plt.show()
