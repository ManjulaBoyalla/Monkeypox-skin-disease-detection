# Monkeypox-skin-disease-detection
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Example model definition
model = Sequential([
    Dense(64, activation='relu', input_shape=(100,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# After training the model, save it
model.save('feature_extractor_model.h5')
-----------------------------------------------------------------------
import joblib
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# Load the models and label binarizer
knn_model = joblib.load('/kaggle/working/knn_model.pkl')
label_binarizer = joblib.load('/kaggle/working/label_binarizer.pkl')
feature_extractor = load_model('/kaggle/working/feature_extractor_model.keras')

# Function to preprocess a single image
def preprocess_image(image_path, target_size=(150, 150)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to predict the class of a single image
def predict_image_class(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)
    
    # Extract features
    features = feature_extractor.predict(img_array)
    features_flat = features.reshape((features.shape[0], -1))
    
    # Predict the class
    prediction = knn_model.predict(features_flat)
    predicted_class = label_binarizer.inverse_transform(prediction)
    
    return predicted_class[0]

# Test the prediction
image_path = '/kaggle/input/monkeypox-skin-image-dataset/Monkeypox Skin Image Dataset - Copy/test/Monkeypox/monkeypox22.png'  # Replace with the path to your image
predicted_class = predict_image_class(image_path)
print(f'The predicted class for the image is: {predicted_class}')
-------------------------------------------------------------------------------------------------
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 1s/step
The predicted class for the image is: Measles
----------------------------------------------------------------------------------------------
pip install seaborn

-----------------------------------------

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model, save_model
from PIL import UnidentifiedImageError

# Function to load images and labels with error handling
def load_images_and_labels(data_dir, target_size=(150, 150)):
    images = []
    labels = []
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = load_img(img_path, target_size=target_size)
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(class_dir)
                except UnidentifiedImageError as e:
                    print(f"Cannot identify image file {img_path}: {e}")
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)

# Load the dataset
train_images, train_labels = load_images_and_labels(r'C:\Users\pmk70\OneDrive\Desktop\New folder\Monkeypox\Backend\Monkeypox Skin Image Dataset\train')
test_images, test_labels = load_images_and_labels(r'C:\Users\pmk70\OneDrive\Desktop\New folder\Monkeypox\Backend\Monkeypox Skin Image Dataset\test')

# Preprocess the images
train_images = preprocess_input(train_images)
test_images = preprocess_input(test_images)

# Load MobileNetV2 model for feature extraction
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

# Extract features
train_features = model.predict(train_images)
test_features = model.predict(test_images)

# Flatten the features
train_features_flat = train_features.reshape((train_features.shape[0], -1))
test_features_flat = test_features.reshape((test_features.shape[0], -1))

# Encode the labels
lb = LabelBinarizer()
train_labels_enc = lb.fit_transform(train_labels)
test_labels_enc = lb.transform(test_labels)

# Decode the one-hot encoded labels to single class labels
train_labels_dec = lb.inverse_transform(train_labels_enc)
test_labels_dec = lb.inverse_transform(test_labels_enc)

# Create the SVM model
svm_model = SVC(kernel='linear', probability=True)

# Train the model
svm_model.fit(train_features_flat, train_labels_dec)

# Save the SVM model and label binarizer
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(lb, 'label_binarizer_m.pkl')

# Save the feature extractor model in HDF5 format
model.save('feature_extractor_model_m.h5')

# Make predictions
test_predictions = svm_model.predict(test_features_flat)

# Calculate accuracy
accuracy = accuracy_score(test_labels_dec, test_predictions)
print(f'Test accuracy: {accuracy:.2f}')

# Compute confusion matrix
cm = confusion_matrix(test_labels_dec, test_predictions, labels=lb.classes_)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=lb.classes_, yticklabels=lb.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
--------------------------------------------------
# Calculate accuracy
accuracy = accuracy_score(test_labels_dec, test_predictions)
print(f'Train accuracy: {accuracy:.2f}')
---------------------------------------------------------------------------------------------------
import joblib
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# Load the models and label binarizer
svm_model = joblib.load('svm_model.pkl')
label_binarizer = joblib.load('label_binarizer_m.pkl')
feature_extractor = load_model('feature_extractor_model_m.h5')

# Function to preprocess a single image
def preprocess_image(image_path, target_size=(150, 150)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to predict the class of a single image
def predict_image_class(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)
    
    # Extract features
    features = feature_extractor.predict(img_array)
    features_flat = features.reshape((features.shape[0], -1))
    
    # Predict the class
    prediction = svm_model.predict(features_flat)
    
    return prediction[0]

# Test the prediction
image_path = '/kaggle/input/monkeypox-skin-image-dataset/Monkeypox Skin Image Dataset - Copy/test/Chickenpox/chickenpox18.png'  # Replace with the path to your image
predicted_class = predict_image_class(image_path)
print(f'The predicted class for the image is: {predicted_class}')
---------------------------------------------------------------------------
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 1s/step
The predicted class for the image is: Chickenpox
