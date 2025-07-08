# ✅ Install dependencies (if needed)
# !pip install -q matplotlib tensorflow opencv-python

# ✅ Import libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import zipfile
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# ✅ Data loading and preprocessing
data = []
labels = []

# Paths
zip_path = "/content/Brain2mer.zip"  # <--- Update this path if needed
extracted_path = "/content/extracted"

# ✅ Extract zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_path)

# ✅ Load and preprocess images
categories = ['yes', 'no']
image_size = 64  # Better resolution than 54x54

for category in categories:
    folder = os.path.join(extracted_path, 'brain_tumor_dataset', category)
    label = 1 if category == 'yes' else 0
    for img_file in os.listdir(folder):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (image_size, image_size))
                data.append(img)
                labels.append(label)

# ✅ Normalize and convert labels
data = np.array(data) / 255.0
labels = to_categorical(np.array(labels))  # Ensure labels are NumPy array

# ✅ Train-test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# ✅ Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(image_size, image_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Train model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# ✅ Evaluate model
loss, acc = model.evaluate(x_test, y_test)
print(f"🔍 Test Accuracy: {acc*100:.2f}%")

# ✅ Predict on a sample image
i = 10
plt.imshow(x_test[i])
pred = np.argmax(model.predict(x_test[i:i+1]))
prediction = "Tumor" if pred == 1 else "No Tumor"
plt.title(f"Predicted: {prediction}")
plt.axis('off')
plt.show()

# ✅ Predict using user-provided index (with safety)
try:
    i = int(input(f"Enter image index to predict (0 to {len(x_test)-1}): "))
    if 0 <= i < len(x_test):
        plt.imshow(x_test[i])
        pred = np.argmax(model.predict(x_test[i:i+1]))
        prediction = "Tumor" if pred == 1 else "No Tumor"
        plt.title(f"Predicted: {prediction}")
        plt.axis('off')
        plt.show()
    else:
        print("⚠️ Invalid index!")
except:
    print("⚠️ Invalid input. Please enter a number.")


# ✅ Loop for continuous predictions (0–50 range)
while True:
    user_input = input(f"\n🔎 Enter image index (0 to {min(50, len(x_test)-1)}) or 'q' to quit: ")

    if user_input.lower() == 'q':
        print("✅ Exiting prediction loop.")
        break

    try:
        i = int(user_input)
        if 0 <= i <= min(50, len(x_test)-1):
            plt.imshow(x_test[i])
            pred = np.argmax(model.predict(x_test[i:i+1]))
            prediction = "Tumor" if pred == 1 else "No Tumor"
            plt.title(f"Predicted: {prediction}")
            plt.axis('off')
            plt.show()
        else:
            print("⚠️ Please enter a valid index between 0 and 50.")
    except ValueError:
        print("⚠️ Invalid input. Please enter a number or 'q' to quit.")
