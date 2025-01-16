import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchvision.models import  ResNet18_Weights
from transformers import ViTForImageClassification
import pandas as pd
from PIL import Image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


data_transforms = transforms.Compose([
    transforms.Resize((236, 236)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


data_dir = '/content/drive/MyDrive/First year/Colab Notebooks/Data/AugmentedTrain'



image_datasets = datasets.ImageFolder(
    root=data_dir,
    transform=data_transforms
)

dataloader = DataLoader(image_datasets, batch_size=32, shuffle=True)
num_epochs=15

model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1.DEFAULT)
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2)
)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

def load_model():
    model_path = '/content/ai_vs_real_model.pth'  # Path to your saved model file
    model.load_state_dict(torch.load(model_path, map_location=device))

def train_model(model,num_epochs,dataloader,criterion,optimizer):
    model.train()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in dataloader:
            # Move the data to the GPU if available
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            # scheduler.step()
            # Track the loss and accuracy
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")
    print("Training Finished!")
    torch.save(model.state_dict(), 'ai_vs_real_model.pth')
    print("Model saved as 'ai_vs_real_model.pth'")



def test_csv(model, data_transforms):
    load_model()
    output_csv = 'unsorted_file.csv'
    test_dir = '/content/drive/MyDrive/First year/Colab Notebooks/Data/Test1/Test' 
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    model.eval()

    class_names = ['AI', 'Real'] 

    with torch.no_grad():
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Id', 'Label'])  

            for image_name in image_files:
                image_path = os.path.join(test_dir, image_name)

                try:
            
                    image = Image.open(image_path).convert('RGB')  # Ensure 3-channel image
                    input_tensor = data_transforms(image).unsqueeze(0).to(device)  # Add batch dimension

                    optimizer.zero_grad()
                    
                    outputs = model(input_tensor)
                    _, preds = torch.max(outputs, 1)
                    label = class_names[preds.item()]

                    writer.writerow([image_name, label])

                except Exception as e:
                    print(f"Error processing {image_name}: {e}")

    print(f"Predictions saved to '{output_csv}'")


    try:
        data = pd.read_csv(output_csv)

        # Clean the 'Id' column: strip spaces, remove '.jpg', and extract numeric parts
        data['Id'] = data['Id'].str.strip().str.replace('.jpg', '', regex=False)
        data['Id_Number'] = data['Id'].str.extract(r'(\d+)').astype(int)

        # Filter IDs within the desired range (1 to 200)
        filtered_data = data[(data['Id_Number'] >= 1) & (data['Id_Number'] <= 200)]

        # Sort the data based on the numeric part of the 'Id'
        sorted_data = filtered_data.sort_values(by='Id_Number', ascending=True).drop(columns=['Id_Number'])

        # Save the sorted data to a new file
        sorted_file_path = 'sorted_file.csv'
        sorted_data.to_csv(sorted_file_path, index=False)

        print(f"File sorted and saved to: {sorted_file_path}")
    except Exception as e:
        print(f"Error sorting the CSV: {e}")


#train_model(model,num_epochs,dataloader,criterion,optimizer)
#test_csv(model, data_transforms)

