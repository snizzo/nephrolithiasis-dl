import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

import os

class PlotUtils:
    '''
    Inherit this class to log training history and generate plots
    '''
    def __init__(self):
        self.training_losses = []
        self.validation_losses = []

        self.train_accuracies = []
        self.val_accuracies = []

        self.prefix_path = "Info_"+self.__class__.__name__+"/"
    
    def getPrefixPath(self):
        return self.prefix_path
    
    def logEpoch(self, train_loss, val_loss, train_acc, val_acc):
        self.training_losses.append(train_loss)
        self.validation_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
    
    def savePlot(self):
        # ensure output directory exists
        out_dir = os.path.dirname(self.getPrefixPath())
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        self.generatePlot(self.getPrefixPath() + "training_plot.png")
        print(f"Training plot saved to {self.getPrefixPath() + 'training_plot.png'}")

    def generatePlot(self, out_path):
        # save a training history plot with loss and accuracy.
        epochs = range(1, len(self.training_losses) + 1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        ax1.plot(epochs, self.training_losses, 'b.-', label='Train Loss')
        ax1.plot(epochs, self.validation_losses, 'r.-', label='Val Loss')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, [a * 100 for a in self.train_accuracies], 'g.-', label='Train Acc (%)')
        # TODO: maybe KidneyCNN doesn't support validation accuracy logging yet, may creates inconsistent plots
        ax2.plot(epochs, [a * 100 for a in self.val_accuracies], 'm.-', label='Val Acc (%)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)