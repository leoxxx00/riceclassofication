import os
import cv2  # For image processing
import pandas as pd
from tqdm import tqdm  # For progress bar
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Specify the directory paths and labels
directories = {
    "Arborio": "/Users/htetaung/Desktop/Desktop - Leon’s MacBook Pro/PJ/Rice_Image_Dataset/Arborio",
    "Basmati": "/Users/htetaung/Desktop/Desktop - Leon’s MacBook Pro/PJ/Rice_Image_Dataset/Basmati",
    "Ipsala": "/Users/htetaung/Desktop/Desktop - Leon’s MacBook Pro/PJ/Rice_Image_Dataset/Ipsala",
    "Jasmine": "/Users/htetaung/Desktop/Desktop - Leon’s MacBook Pro/PJ/Rice_Image_Dataset/Jasmine",
    "Karacadag": "/Users/htetaung/Desktop/Desktop - Leon’s MacBook Pro/PJ/Rice_Image_Dataset/Karacadag"
}

# Initialize a list to hold image data and labels
image_data = []
total_images = 0  # To count total number of image files

# Supported image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Function to extract features from an image
def extract_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    # Example feature extraction: image shape (height, width, channels)
    height, width, channels = image.shape

    # Example feature: Flattened color histogram for the image (simple feature)
    color_histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    color_histogram = color_histogram.flatten()  # Make it a 1D array

    return height, width, channels, color_histogram[:5]  # Save first 5 values for brevity

# Track the overall progress
total_files = sum(len(files) for directory in directories.values() if os.path.exists(directory) for files in [os.listdir(directory)])
progress_bar = tqdm(total=total_files, desc="Processing images")

# Iterate over each directory and assign labels
for label, directory in directories.items():
    # Check if the directory exists
    if os.path.exists(directory):
        print(f"Processing directory: {directory}")
        # List all files in the directory
        files = os.listdir(directory)

        # Iterate over each file
        for file in files:
            file_path = os.path.join(directory, file)

            # Check if the file is an image
            if any(file.lower().endswith(ext) for ext in image_extensions):
                total_images += 1  # Increment the image file count
                # Extract image features
                features = extract_features(file_path)

                if features is not None:
                    height, width, channels, hist = features
                    # Append the data (filename, file path, features, and label) to the list
                    image_data.append({
                        'file_name': file,
                        'file_path': file_path,
                        'height': height,
                        'width': width,
                        'channels': channels,
                        'color_histogram': hist,  # Example of saving histogram
                        'label': label  # Add the label based on the directory name
                    })
                else:
                    # Print an error if the image could not be read
                    print(f"Warning: Could not extract features from {file_path}")
            else:
                print(f"Skipping non-image file: {file_path}")

            # Update the progress bar
            progress_bar.update(1)
    else:
        print(f"The specified directory does not exist: {directory}")

# Close the progress bar
progress_bar.close()

# Print the total number of image files processed
print(f"Total number of image files processed: {total_images}")

# Convert the list to a DataFrame
df = pd.DataFrame(image_data)

# Feature preparation: Flatten the color histogram into columns and prepare the labels
color_hist_cols = [f"color_hist_{i}" for i in range(5)]
df[color_hist_cols] = pd.DataFrame(df['color_histogram'].tolist(), index=df.index)
df = df.drop(columns=['color_histogram', 'file_name', 'file_path', 'height', 'width', 'channels'])

# Split data into features (X) and labels (y)
X = df[color_hist_cols]
y = df['label']

# Now split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Print the accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Optionally, save the train and test sets to new CSV files
X_train['label'] = y_train
X_test['label'] = y_test
X_train.to_csv("train_image_features.csv", index=False)
X_test.to_csv("test_image_features.csv", index=False)

print("Train and test CSV files have been saved.")
