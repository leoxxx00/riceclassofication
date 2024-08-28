import cv2
import numpy as np
import os
import pandas as pd

# Load the image using OpenCV
image_path = '/Users/htetaung/Desktop/Desktop - Leon’s MacBook Pro/PJ/Rice_Image_Dataset/Arborio/Arborio (1).jpg'

# Check if the file exists
if os.path.exists(image_path):
    # Attempt to load the image
    image = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if image is not None:
        # Convert the image to grayscale (or you can skip this step for RGB)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert the image into a matrix
        image_matrix = np.array(image_gray)

        # Print the matrix values
        print("Image matrix (grayscale):")
        print(image_matrix)

        # Convert the matrix to a DataFrame for CSV export
        df = pd.DataFrame(image_matrix)

        # Save the DataFrame to a CSV file
        csv_path = 'image_matrix.csv'
        df.to_csv(csv_path, index=False)

        print(f"Image matrix has been saved as {csv_path}")
    else:
        print("Error: Failed to load the image. Please check the file format or path.")
else:
    print(f"Error: File not found at {image_path}")
