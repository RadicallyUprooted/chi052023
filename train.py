import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, ToPILImage, RandomAffine
import torchvision.models as models
import torch.cuda.amp as amp


def extract(input_path, output_path):
    '''Extracts digits and uppercase letters from balanced EMNIST dataset and puts them into a new file.
        Also removes capital letters I, O, Q and remaps labels accordingly. Requires EMNIST csv file.'''
    initial_labels = list(range(36))
    updated_labels = list(filter(lambda x: x not in (18, 24, 26), initial_labels))
    mapping = {l: i for (i, l) in enumerate(updated_labels)}

    with open(input_path, "r") as r, open(output_path, "w", newline='') as w:
        for line in r:
            values = line.strip().split(',')
            label = int(values[0])
            if (label <= 35) and (label not in (26, 24, 18)):
                new_label = mapping[label]
                w.write(f"{new_label},{','.join(values[1:])}\n")

class Net(nn.Module):
    '''A VGG-like architecture.'''
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()
        self.fc = nn.Identity()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ResNet18(nn.Module):
    '''Redefined input layer.'''
    def __init__(self):
        super(ResNet18, self).__init__()

        self.resnet = models.resnet18()
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.resnet.fc = nn.Identity()

    def forward(self, x):

        x = self.resnet(x)

        return x

class Ensemble(nn.Module):
    '''This type of combination showed itself in a good light during experiments.
    If a checkpoint is being loaded, requires ResNet18() as model_1 and Net() as model_2'''
    def __init__(self, model_1, model_2):
        super(Ensemble, self).__init__()

        self.model_1 = model_1
        self.model_2 = model_2
        self.fc = nn.Linear(512, 33)

    def forward(self, x):

        x1 = self.model_1(x)
        x2 = self.model_2(x)
        output = torch.add(x1, x2)
        output = self.fc(output)

        return output

class EMNIST(Dataset):
    '''Data preparation for a loader.'''
    def __init__(self, path, train = True, transform = None):

        self.data = []
        self.transform = transform
        self.train = train

        with open(path, 'r') as f:
            for line in f:
                values = line.strip().split(',')
                label = int(values[0])
                image = torch.tensor(list(map(int, values[1:])), dtype=torch.float32).view(1, 28, 28) / 255.0
                self.data.append((image, label))

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        image, label = self.data[idx] 
        if self.train and self.transform:
            image = self.transform(image)

        return image, label

def train(model, train_loader, test_loader, epochs, criterion, optimizer, scheduler):

    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        train_correct = 0.0

        model.train()
        for batch in train_loader:

            image, target = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()

            with amp.autocast():
                output = model(image.to(device))
                loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * image.size(0)
            train_pred =  torch.max(output, 1)[1]
            train_correct += (train_pred == target).sum()

            del image, target, output, loss

        test_loss = 0.0
        test_correct = 0.0
        model.eval()
        with torch.no_grad():
            for batch in test_loader:

                image, target = batch[0].to(device), batch[1].to(device)

                with amp.autocast():
                    output = model(image)
                    loss = criterion(output, target)

                test_loss += loss.item() * image.size(0)
                test_pred = torch.max(output, 1)[1]
                test_correct += (test_pred == target).sum()

                del image, target, output, loss

        scheduler.step()

        print(f'''Epoch {epoch}/{epochs}
Train Loss: {train_loss / len(train_loader.dataset):.4f} -- Accuracy: {train_correct / len(train_loader.dataset):.4f}
Test Loss: {test_loss / len(test_loader.dataset):.4f} -- Accuracy: {test_correct / len(test_loader.dataset):.4f}''')

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"models\model_{epoch}.pth")


def main():

    resnet18 = ResNet18()
    vgglike = Net()
    ensemble_model = Ensemble(resnet18, vgglike)
    
    resnet18.to(device)
    vgglike.to(device)
    ensemble_model.to(device)

    transform = Compose([ToPILImage(), RandomAffine((-10, 10), translate=(0.1, 0.1), shear=(-0.2, 0.2), scale=(0.9, 1.1)), ToTensor()])
    
    train_dataset = EMNIST('emnist\emnist-train.csv', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    test_dataset = EMNIST('emnist\emnist-test.csv', train=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    epochs = 50

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    train(ensemble_model, train_loader, test_loader, epochs, criterion, optimizer, scheduler)

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    main()


