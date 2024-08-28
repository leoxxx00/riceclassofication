import os
import cv2  # For image processing
import pandas as pd
from tqdm import tqdm  # For progress bar

# Specify the directory paths
directories = [
    "/Users/htetaung/Desktop/Desktop - Leon’s MacBook Pro/PJ/Rice_Image_Dataset/Arborio",
    "/Users/htetaung/Desktop/Desktop - Leon’s MacBook Pro/PJ/Rice_Image_Dataset/Basmati",
    "/Users/htetaung/Desktop/Desktop - Leon’s MacBook Pro/PJ/Rice_Image_Dataset/Ipsala",
    "/Users/htetaung/Desktop/Desktop - Leon’s MacBook Pro/PJ/Rice_Image_Dataset/Jasmine",
    "/Users/htetaung/Desktop/Desktop - Leon’s MacBook Pro/PJ/Rice_Image_Dataset/Karacadag"
]

# Initialize a list to hold image data
image_data = []

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
total_files = sum(len(files) for directory in directories if os.path.exists(directory) for files in [os.listdir(directory)])
progress_bar = tqdm(total=total_files, desc="Processing images")

# Iterate over each directory
for directory in directories:
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
                # Extract image features
                features = extract_features(file_path)

                if features is not None:
                    height, width, channels, hist = features
                    # Append the data (filename, file path, and features) to the list
                    image_data.append({
                        'file_name': file,
                        'file_path': file_path,
                        'height': height,
                        'width': width,
                        'channels': channels,
                        'color_histogram': hist  # Example of saving histogram
                    })
                else:
                    # Print an error if the image could not be read, especially focusing on .png files
                    if file.lower().endswith('.png'):
                        print(f"Error: Could not extract features from .png file: {file_path}")
                    else:
                        print(f"Warning: Could not extract features from {file_path}")
            else:
                print(f"Skipping non-image file: {file_path}")

            # Update the progress bar
            progress_bar.update(1)
    else:
        print(f"The specified directory does not exist: {directory}")

# Close the progress bar
progress_bar.close()

# Convert the list to a DataFrame and save as a CSV file
df = pd.DataFrame(image_data)
df.to_csv("image_features.csv", index=False)

print("Feature extraction complete and saved to 'image_features.csv'.")
