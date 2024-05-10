from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import os
from imutils import paths
import cv2 
import matplotlib.pyplot as plt
import seaborn as sns

# Function to import images and labels
def import_images_data(samples_directory):
    image_pixels = []
    image_labels = []

    for image_path in samples_directory:
        sample_image = cv2.imread(image_path)
        sample_label = image_path.split(os.path.sep)[-2]
        sample_image = cv2.resize(sample_image, (32, 32), interpolation=cv2.INTER_AREA)
        image_pixels.append(sample_image)
        image_labels.append(sample_label)

    return (np.array(image_pixels), np.array(image_labels))

# Directory containing images
#path_to_sample_images = 'raw-img'
path_to_sample_images = '/content/drive/My Drive/colab_data/animals'
samples_directory = list(paths.list_images(path_to_sample_images))
image_pixels, image_labels = import_images_data(samples_directory)

# Reshaping image pixels
image_pixels = image_pixels.reshape((image_pixels.shape[0], 3072))

# Encoding labels
encod_image_labels = LabelEncoder()
image_labels = encod_image_labels.fit_transform(image_labels)

# Splitting dataset: 60% training, 15% validation, 25% test
print("Splitting Data...")
trainig_image_pixels, temp_image_pixels, training_image_labels, temp_image_labels = train_test_split(
    image_pixels, image_labels, test_size=0.4, random_state=42)
validation_image_pixels, test_image_pixels, validation_image_labels, test_image_labels = train_test_split(
    temp_image_pixels, temp_image_labels, test_size=0.625, random_state=42)  # 0.625 * 0.4 = 0.25

# Finding the best n_neighbors using the validation set
print("Starting Classification...")
validation_accuracies = []
test_accuracies = []
n_neighbors_range = range(1, 11)

for n_neighbors in n_neighbors_range:
    print(f"Trying n_neighbors = {n_neighbors}...")
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(trainig_image_pixels, training_image_labels)

    # Validation predictions and accuracy
    val_predictions = knn_classifier.predict(validation_image_pixels)
    val_accuracy = accuracy_score(validation_image_labels, val_predictions)
    validation_accuracies.append(val_accuracy)

    # Test predictions and accuracy
    test_predictions = knn_classifier.predict(test_image_pixels)
    test_accuracy = accuracy_score(test_image_labels, test_predictions)
    test_accuracies.append(test_accuracy)

    print(f"Validation Accuracy: {val_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")

# Plotting n_neighbors vs accuracy for validation and test
plt.figure(figsize=(10, 7))
plt.plot(n_neighbors_range, validation_accuracies, label='Validation Accuracy', marker='o')
plt.plot(n_neighbors_range, test_accuracies, label='Test Accuracy', marker='o')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.title('n_neighbors vs Accuracy for Validation and Test Sets')
plt.xticks(n_neighbors_range)
plt.legend()
plt.grid(True)
plt.show()
