# using preexisting model to extract features
# in this file EfficientNetBX (for now B0/B2/B5/B7) is used

import torch
import torch.nn as nn

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

from features import FeatureVisualizer
from PlotUtils import PlotUtils

class KidneyEfficientNet(FeatureVisualizer, PlotUtils):
    def __init__(self, data, fine_tune="none"):
        FeatureVisualizer.__init__(self)
        PlotUtils.__init__(self)

        self.data = data
        self.fine_tune = fine_tune
        self.model = None

        self.initialize()

    def getModel(self):
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() before getModel().")
        return self.model
    
    def getData(self):
        return self.data

    def initialize(self):
        # download will be performed automatically from pytorch servers
        print("[INFO]: Loading EfficientNetB0 model...")
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        if self.fine_tune == "all":
            print("[INFO]: Fine-tuning all layers...")
            for params in model.parameters():
                params.requires_grad = True
        elif self.fine_tune == "partial":
            print("[INFO]: Freezing first 5 blocks, training the rest...")
            # Freeze the first 5 feature blocks
            for i in range(5):  # blocks 0-4 frozen
                for param in model.features[i].parameters():
                    param.requires_grad = False
        elif self.fine_tune == "none":
            print("[INFO]: Freezing hidden layers...")
            # this one trains only the lastfully connected layer
            # therefore not requiring the gradient
            for name, params in model.named_parameters():
                print("Freezing layer:", name, params.shape)
                params.requires_grad = False
        else:
            raise ValueError("Invalid fine_tune option. Choose from 'all', 'partial', or 'none'.")
        
        # change the final classification head
        # out features is 2 for binary classification (NL vs not NL)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=2)

        # saving for internal use
        self.model = model

        # returning just for redundancy
        return model
    
    def train(self, num_epochs=10):
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() before train().")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        model = self.model
        model = model.to(device)

        train_loader = self.data.getTrainLoader()
        val_loader = self.data.getValLoader()

        for epoch in range(num_epochs):
            model.train()  # activate dropout etc...

            train_loss = 0.0
            running_acc = 0.0

            for images, labels in train_loader:

                # moves data to GPU
                images = images.to(device)
                labels = labels.to(device)

                # set gradients to zero
                optimizer.zero_grad()

                # forward pass
                outputs = self.model(images)

                # compute loss
                loss = criterion(outputs, labels)

                # accuracy
                _, preds = torch.max(outputs, 1)
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
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    v_loss = criterion(outputs, labels)

                    val_loss += v_loss.item()
                    _, val_preds = torch.max(outputs, 1)
                    v_correct = (val_preds == labels).sum().item()
                    batch_val_acc = v_correct / labels.size(0)
                    val_acc += batch_val_acc
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_train_acc = running_acc / len(train_loader)
            avg_val_acc = val_acc / len(val_loader)

            # needed to print plots later
            self.logEpoch(avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc)

            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} "
                  f"Train Acc: {avg_train_acc:.4f} "
                  f"Val Loss: {avg_val_loss:.4f} "
                  f"Val Acc: {avg_val_acc:.4f}")
        
        self.savePlot()
            
        return model
