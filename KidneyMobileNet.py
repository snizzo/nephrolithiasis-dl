import torch
import torch.nn as nn

from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
# optional: mobilenet_v3_small, MobileNet_V3_Small_Weights

from features import FeatureVisualizer
from PlotUtils import PlotUtils

class KidneyMobileNet(FeatureVisualizer, PlotUtils):
    def __init__(self, data, fine_tune=False):
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
        print("[INFO]: Loading MobileNetV3-Large model...")
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

        if self.fine_tune:
            print("[INFO]: Fine-tuning all layers...")
            for p in model.parameters():
                p.requires_grad = True
        else:
            # TODO: doesn't actually deliver acceptable performance
            print("[INFO]: Freezing hidden layers (train only classifier)...")
            for p in model.parameters():
                p.requires_grad = False

        # last layer is needed to binary classify,
        # out feats is 2 cause becaues we use CrossEntropyLoss
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features=in_features, out_features=2)

        # set last layer to be trainable
        for p in model.classifier.parameters():
            p.requires_grad = True

        self.model = model
        return model

    def train(self, num_epochs=10, lr=1e-3):
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() before train().")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        model = self.model.to(device)

        criterion = nn.CrossEntropyLoss()

        # optimize only trainable parameters
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params_to_optimize, lr=lr)

        train_loader = self.data.getTrainLoader()
        val_loader = self.data.getValLoader()

        for epoch in range(num_epochs):
            model.train()

            train_loss = 0.0
            train_acc = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                # accuracy
                _, preds = torch.max(outputs, 1)
                correct = (preds == labels).sum().item()

                # batch accuracy
                batch_acc = correct / labels.size(0)
                train_acc += batch_acc

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            val_acc = 0.0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_acc += (preds == labels).float().mean().item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_train_acc = train_acc / len(train_loader)
            avg_val_acc = val_acc / len(val_loader)

            self.logEpoch(avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc)

            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {avg_train_loss:.4f} "
                f"Train Acc: {avg_train_acc:.4f} "
                f"Val Loss: {avg_val_loss:.4f} "
                f"Val Acc: {avg_val_acc:.4f}"
            )
        
        self.savePlot()

        return model
