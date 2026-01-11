import torch
import torch.nn as nn

# plotting for training history
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from features import FeatureVisualizer
from PlotUtils import PlotUtils

class KidneyCNN(nn.Module):
    def __init__(self):
        super(KidneyCNN, self).__init__()

        # one input channel because the image is grayscale
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        #TODO: usare nn.AdaptiveAvgPool2d((1, 1)) per resolution indipendent processing
    
        # sezione fully connected
        self.fc1 = nn.Linear(25088, 128)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.block1(x) # da 224 a 112
        x = self.block2(x) # da 112 a 56
        x = self.block3(x) # da 56  a 28

        x = torch.flatten(x, 1)  # (batch, 25088)

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)

        x = self.fc3(x)
        #x = self.sigmoid(x)
        return x

class KidneyCNNTrainer(FeatureVisualizer, PlotUtils):
    def __init__(self, data):
        FeatureVisualizer.__init__(self)
        PlotUtils.__init__(self)

        self.model = KidneyCNN()
        self.data = data
    
    def getModel(self):
        return self.model
    
    def getData(self):
        return self.data
    
    def saveModel(self, filename="kidney_cnn_model.pth"):
        torch.save(self.model.state_dict(), self.getPrefixPath() + filename)

    def train(self, num_epochs=10):

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-4)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        self.model.to(device)

        train_loader = self.data.getTrainLoader()
        val_loader = self.data.getValLoader()

        for epoch in range(num_epochs):
            self.model.train()  # activate dropout etc...

            train_loss = 0.0
            running_acc = 0.0

            for images, labels in train_loader:

                # moves data to GPU
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)  
                # float because BCELoss wants float
                # unsqueeze(1) because labels should have form (batch, 1)

                # set gradients to zero
                optimizer.zero_grad()

                # forward pass
                outputs = self.model(images)

                # compute loss
                loss = criterion(outputs, labels)

                # accuracy
                # get binary predictions
                preds = (outputs >= 0.5).float()

                # compare with the labels
                correct = (preds == labels).sum().item()

                # batch accuracy
                batch_acc = correct / labels.size(0)
                running_acc += batch_acc

                # backward pass
                loss.backward()

                # update weights
                optimizer.step()

                train_loss += loss.item()

            # validation pass
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for images, labels in val_loader:

                    images = images.to(device)
                    labels = labels.float().unsqueeze(1).to(device)

                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()

            # average losses / accuracy for the epoch
            avg_train_loss = train_loss / len(train_loader)
            val_loss /= len(val_loader)
            epoch_acc = running_acc / len(train_loader)

            self.logEpoch(avg_train_loss, val_loss, epoch_acc, epoch_acc)

            print(f"Epoch {epoch+1}/{num_epochs}  "
                f"Train Loss: {avg_train_loss:.4f}  "
                f"Val Loss: {val_loss:.4f}")
            print(f"Train Accuracy: {epoch_acc*100:.2f}%")

        # after training, plot history
        self.savePlot()