import os

# Specify the directory paths
directories = [
    "/Users/htetaung/Desktop/Desktop - Leon’s MacBook Pro/PJ/Rice_Image_Dataset/Arborio",
    "/Users/htetaung/Desktop/Desktop - Leon’s MacBook Pro/PJ/Rice_Image_Dataset/Basmati",
    "/Users/htetaung/Desktop/Desktop - Leon’s MacBook Pro/PJ/Rice_Image_Dataset/Ipsala",
    "/Users/htetaung/Desktop/Desktop - Leon’s MacBook Pro/PJ/Rice_Image_Dataset/Jasmine",
    "/Users/htetaung/Desktop/Desktop - Leon’s MacBook Pro/PJ/Rice_Image_Dataset/Karacadag"
]

# Iterate over each directory
for directory in directories:
    # Check if the directory exists
    if os.path.exists(directory):
        print(f"Files in directory: {directory}")
        # List all files in the directory
        files = os.listdir(directory)

        # Print the filenames
        for file in files:
            print(file)
        print()  # Print an empty line for better readability
    else:
        print(f"The specified directory does not exist: {directory}")
