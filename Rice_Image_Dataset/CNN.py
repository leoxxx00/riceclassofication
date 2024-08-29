import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm  # Progress bar

# Specify the directory paths and labels
directories = {
    "Arborio": "/Users/htetaung/Desktop/Desktop /PJ/Rice_Image_Dataset/Arborio",
    "Basmati": "/Users/htetaung/Desktop/Desktop /PJ/Rice_Image_Dataset/Basmati",
    "Ipsala": "/Users/htetaung/Desktop/Desktop /PJ/Rice_Image_Dataset/Ipsala",
    "Jasmine": "/Users/htetaung/Desktop/Desktop /PJ/Rice_Image_Dataset/Jasmine",
    "Karacadag": "/Users/htetaung/Desktop/Desktop /PJ/Rice_Image_Dataset/Karacadag"
}

# Supported image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Custom Dataset class
class RiceDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Prepare the data
file_paths = []
labels = []

# Collect file paths and labels
for label, directory in directories.items():
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                file_paths.append(os.path.join(directory, file))
                labels.append(label)
    else:
        print(f"Directory does not exist: {directory}")

# Add debugging prints to check if we have found any files
print(f"Number of files found: {len(file_paths)}")
if len(file_paths) > 0:
    print(f"Sample file paths: {file_paths[:5]}")  # Print the first 5 file paths
    print(f"Sample labels: {labels[:5]}")  # Print the first 5 labels
else:
    raise ValueError("No image files were found. Please check your directory paths and image extensions.")

# Encode the labels into integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split into training and testing sets
train_paths, test_paths, train_labels, test_labels = train_test_split(
    file_paths, encoded_labels, test_size=0.2, random_state=42
)

print(f"Train size: {len(train_paths)}, Test size: {len(test_paths)}")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Downsample the images to 64x64
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets and data loaders
train_dataset = RiceDataset(train_paths, train_labels, transform=transform)
test_dataset = RiceDataset(test_paths, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # Adjust the input size based on the new image size
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)  # Adjust the flatten size based on the new image size
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
num_classes = len(label_encoder.classes_)
model = SimpleCNN(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Print the model architecture
print(model)

# Training loop
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Variables to store loss and accuracy for plotting
train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    with tqdm(train_loader, unit="batch") as tepoch:
        for images, labels in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}")
            images, labels = images.to(device), labels.to(device)

            # Print input shape (only for the first batch to avoid clutter)
            if epoch == 0 and total == 0:
                print(f"Input shape: {images.shape}")

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            tepoch.set_postfix(loss=running_loss / (total / images.size(0)))

    train_accuracy = correct / total
    train_losses.append(running_loss / len(train_loader))
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, Accuracy: {train_accuracy}")

    # Test after each epoch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    test_accuracies.append(test_accuracy)
    print(f"Test Accuracy after Epoch {epoch + 1}: {test_accuracy}")

# Plot and save loss and accuracy curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('loss_accuracy_curves.png')
plt.show()
print("Loss and accuracy curves saved as 'loss_accuracy_curves.png'")

# Evaluate the model and print the classification report
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

# Compute and plot confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')
plt.show()
print("Confusion matrix saved as 'confusion_matrix.png'")

# Save the model (optional)
torch.save(model.state_dict(), 'rice_cnn_model.pth')
print("Model saved as 'rice_cnn_model.pth'")
