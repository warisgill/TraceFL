import torch
from datasets import load_dataset
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score
from transformers import TrainingArguments, Trainer
import numpy as np
from PIL import Image
import logging

# Load the accuracy metric
from datasets import load_metric
metric = load_metric("accuracy")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc}

# Enable logging
logging.basicConfig(level=logging.INFO)

# Load dataset
ds = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", "image-classification", trust_remote_code=True, num_proc=16)

# Use a subset of the dataset for quick testing
ds = ds.shuffle(seed=42)
small_train_ds = ds['train'].select(range(1*1024))  # Use first 1024 samples for training
small_eval_ds = ds['test'].select(range(1024))  # Use first 1024 samples for evaluation

# Define normalization and transformations
_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

# Preprocessing function
def preprocess_function(examples):
    processed_images = []
    for img in examples["image"]:
        img = img.convert("L")  # Ensure the image is in grayscale
        img = img.convert("RGB")  # Ensure the image is in RGB
        img_tensor = _transforms(img)
        processed_images.append(img_tensor)
    examples["pixel_values"] = processed_images
    labels = [label[0] if isinstance(label, (list, tuple)) else label for label in examples['labels']]
    examples['labels'] = torch.tensor(labels, dtype=torch.long)
    del examples['image']
    return examples

# Prepare the dataset
small_train_ds = small_train_ds.with_transform(preprocess_function)
small_eval_ds = small_eval_ds.with_transform(preprocess_function)

# Define labels
labels = ds['train'].features['labels'].feature.names
# print(f'labels {labels}')

# Custom dataset class for PyTorch
class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item['pixel_values'], item['labels']

train_dataset = CustomDataset(small_train_ds)
eval_dataset = CustomDataset(small_eval_ds)

# Calculate class weights
label_list = [item['labels'].item() for item in small_train_ds]
class_counts = np.bincount(label_list)
print (f'class_counts {class_counts}')
class_weights = 1. / class_counts
sample_weights = np.array([class_weights[label] for label in label_list])

# Create a WeightedRandomSampler
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# DataLoader with WeightedRandomSampler
train_dataloader = DataLoader(train_dataset, batch_size=32)
eval_dataloader = DataLoader(eval_dataset, batch_size=256)

# Initialize ResNet model
model = models.densenet121(pretrained=True)

# model.features[0] = torch.nn.Conv2d(
#                 1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
#             )

features = model.classifier.in_features
model.classifier = torch.nn.Linear(features, len(labels))

# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, len(labels))

# # Modify the first convolution layer to accept 1 channel
# model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)



print(class_weights)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# test torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.5e-3)

num_epochs = 25

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader)}')

print('Finished Training')


# # Create a DataLoader for the validation/test set
# val_dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# Evaluate the model
model.eval()  # Set model to evaluation mode
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy: {:.2f}%'.format(100 * correct / total))


