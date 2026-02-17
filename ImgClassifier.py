import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
from torchsummary import summary
from IPython.display import Image as IMG
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMG('image_classification_flowchart.png')

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=(5, 50), translate=(0.1, 0.3), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

batch_size = 16

train_percentage = 90
train_size = len(full_trainset)
train_size = int(train_size * (train_percentage / 100))

indices = list(range(0, len(full_trainset)))

print(indices)
random.seed(42)
random.shuffle(indices)
print(indices)

train_indices = indices[:train_size]

small_trainset = Subset(full_trainset, train_indices)

trainloader = torch.utils.data.DataLoader(small_trainset, batch_size=batch_size, shuffle=True, num_workers=0)

validation_percentage = 10
valid_size = len(full_trainset)
validation_size = int(valid_size * (validation_percentage / 100))

valid_indices = indices[train_size:train_size+valid_size]

small_validationset = Subset(full_trainset, valid_indices)

validloader = torch.utils.data.DataLoader(small_validationset, batch_size=batch_size, shuffle=False, num_workers=0)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
(test_size, _, _, _) = testset.data.shape

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("Number of Images in the Training set:\n",len(small_trainset))
print("Number of Images in the Validation set:\n",len(small_validationset))
print("Test set shape (Number of Images, Height, Width, Number of channels):\n", testset.data.shape)
print("Available classes: ", classes)


def plot_images(dataset):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        img = img / 2 + 0.5
        npimg = img.numpy()
        img = np.transpose(npimg, (1, 2, 0))
        figure.add_subplot(rows, cols, i)
        plt.title(classes[label])
        plt.axis("off")
        plt.imshow(img)
    plt.show()

plot_images(full_trainset)

plot_images(testset)


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, num_classes),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


convnet = ConvNet(10).to(device)

summary(convnet, (3, 32, 32))

writer = SummaryWriter('runs/CIFAR10')

dataiter = iter(trainloader)
images, labels = next(dataiter)
images = images / 2 + 0.5

img_grid = torchvision.utils.make_grid(images)

writer.add_image('CIFAR10_batch', img_grid)


writer.add_graph(convnet, images.to(device))
writer.close()


criteria = nn.CrossEntropyLoss()

optimizer = optim.SGD(convnet.parameters(), lr=0.001, momentum=0.9)

n_epochs = 10
hist_train=[]
hist_valid=[]
best_loss=float('inf')
best_model_wts = copy.deepcopy(convnet.state_dict())
early_stop_tolerant_count=0
early_stop_tolerant=10

print("Start Training ", device)

for epoch in range(n_epochs):
    train_loss = 0.0
    convnet.train()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = convnet(inputs)
        loss = criteria(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(trainloader)
    hist_train.append(train_loss)

    convnet.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = convnet(inputs)
            loss = criteria(outputs, labels)
            valid_loss += loss.item()

    valid_loss /= len(validloader)
    hist_valid.append(valid_loss)

    early_stop_tolerant_count += 1
    if valid_loss < best_loss:
        early_stop_tolerant_count = 0
        best_loss = valid_loss
        best_model_wts = copy.deepcopy(convnet.state_dict())

    if early_stop_tolerant_count >= early_stop_tolerant:
        break

    print(f"Epoch {epoch:04d}: train loss {train_loss:.4f}, valid loss {valid_loss:.4f}")

print('Finished Training')

plt.figure(figsize=(10,6))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss by Epoch')

plt.plot(hist_train)
plt.plot(hist_valid)
plt.legend(['train', 'valid'])


PATH = './convnet.pth'
torch.save(convnet.state_dict(), PATH)


dataiter = iter(validloader)
data = next(dataiter)
images = data[0].to(device)
labels = data[1].to(device)

convnet = ConvNet().to(device)
convnet.load_state_dict(torch.load(PATH, weights_only=True))

outputs = convnet(images)

_, predicted = torch.max(outputs, 1)

grid_img = torchvision.utils.make_grid(images.cpu(), nrow=4, padding=2, normalize=True)

plt.figure(figsize=(10, 10))

plt.imshow(np.transpose(grid_img.numpy(), (1, 2, 0)))

for i in range(len(images)):
    plt.text(i % 4 * 32 + 5, (i // 4) * 32 + 5, f'True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}',
             color="white", fontsize=8, ha="center", va="top", bbox=dict(facecolor='black', alpha=0.6))

plt.axis('off')
plt.show()

"""#### **Accuracy on Test set**"""

correct = 0
total = 0
labels_total=[]
prediction_total=[]
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = convnet(inputs).to(device)
        _, predicted = torch.max(outputs.data, 1)
        labels_total=np.append(labels_total,labels.cpu().numpy(), axis=0)
        prediction_total=np.append(prediction_total,predicted.cpu().numpy(), axis=0)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


cm=confusion_matrix(labels_total, prediction_total)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp = disp.plot(cmap="gist_heat")


correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in validloader:
        images, labels = data
        outputs = convnet(images.to(device))
        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                   accuracy))
