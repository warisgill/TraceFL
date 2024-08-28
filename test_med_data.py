from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator
import os

data_flag = 'pathmnist'
# data_flag = 'breastmnist'
download = True

NUM_EPOCHS = 3
BATCH_SIZE = 128
lr = 0.001

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

print(type(DataClass))


# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load the data
# train_dataset = DataClass(split='train', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download, size=224, mmap_mode='r')
# pil_dataset = DataClass(split='train', download=download)

# help(DataClass)

# Directory where the images will be saved
save_dir = 'filtered_images'
os.makedirs(save_dir, exist_ok=True)

labels_to_save = [4,7]

l4 = 0 
l7 = 0

# Iterate through the dataset
for idx, (image, label) in enumerate(test_dataset):
    # Check if the label is either 2 or 4
    if label in labels_to_save:
        # Convert tensor to a PIL image
        pil_image = transforms.ToPILImage()(image)
        # Define the path to save the image
        image_path = os.path.join(save_dir, f'image_{idx}_label_{label.item()}.png')
        # Save the image
        pil_image.save(image_path)
        l4 += 1
        l7 += 1
        print(f"Image {idx} with label {label.item()} saved to {image_path}")
    if l4 == 20 and l7 == 20:
        break

print(f"Images with labels 2 and 4 have been saved to {save_dir}")




# # encapsulate data into dataloader form
# train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
# test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)


# print(train_dataset)
# print("===================")
# print(test_dataset)

# # define a simple CNN model

# class Net(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(Net, self).__init__()

#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels, 16, kernel_size=3),
#             nn.BatchNorm2d(16),
#             nn.ReLU())

#         self.layer2 = nn.Sequential(
#             nn.Conv2d(16, 16, kernel_size=3),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))

#         self.layer3 = nn.Sequential(
#             nn.Conv2d(16, 64, kernel_size=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU())
        
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU())

#         self.layer5 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))

#         self.fc = nn.Sequential(
#             nn.Linear(64 * 4 * 4, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes))

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

# model = Net(in_channels=n_channels, num_classes=n_classes)
    
# # define loss function and optimizer
# if task == "multi-label, binary-class":
#     criterion = nn.BCEWithLogitsLoss()
# else:
#     criterion = nn.CrossEntropyLoss()
    
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)


# # train

# for epoch in range(NUM_EPOCHS):
#     train_correct = 0
#     train_total = 0
#     test_correct = 0
#     test_total = 0
    
#     model.train()
#     for inputs, targets in tqdm(train_loader):
#         # forward + backward + optimize
#         optimizer.zero_grad()
#         outputs = model(inputs)
        
#         if task == 'multi-label, binary-class':
#             targets = targets.to(torch.float32)
#             loss = criterion(outputs, targets)
#         else:
#             targets = targets.squeeze().long()
#             loss = criterion(outputs, targets)
        
#         loss.backward()
#         optimizer.step()




# # evaluation

# def test(split):
#     model.eval()
#     y_true = torch.tensor([])
#     y_score = torch.tensor([])
    
#     data_loader = train_loader_at_eval if split == 'train' else test_loader

#     with torch.no_grad():
#         for inputs, targets in data_loader:
#             outputs = model(inputs)

#             if task == 'multi-label, binary-class':
#                 targets = targets.to(torch.float32)
#                 outputs = outputs.softmax(dim=-1)
#             else:
#                 targets = targets.squeeze().long()
#                 outputs = outputs.softmax(dim=-1)
#                 targets = targets.float().resize_(len(targets), 1)

#             y_true = torch.cat((y_true, targets), 0)
#             y_score = torch.cat((y_score, outputs), 0)

#         y_true = y_true.numpy()
#         y_score = y_score.detach().numpy()
        
#         evaluator = Evaluator(data_flag, split)
#         metrics = evaluator.evaluate(y_score)
    
#         print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))

        
# print('==> Evaluating ...')
# test('train')
# test('test')

