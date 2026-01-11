import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms

from torch.utils.data import random_split

from data import Data as MyData
from KidneyCNN import KidneyCNN, KidneyCNNTrainer
# ignoring experimental networks for now...
# from KidneyVGG import KidneyVGG, KidneyVGGTrainer
# from KidneyCNNE import KidneyCNNE, KidneyCNNETrainer

# pretrained models
from KidneyEfficientNet import KidneyEfficientNet
from KidneyMobileNet import KidneyMobileNet

import sys
import argparse

# some example of command line calls:
#
# pipenv shell
#
# python main.py --train cnn --data traindata --epochs 10 --batch_size 64 --train_split 0.8 --device cuda
# python main.py --train vgg --data traindata --epochs 15 --batch_size 64 --device cuda
# python main.py --train cnne --data traindata --epochs 20 --batch_size 64 --device cuda --tanh
# --tanh and --sigmoid are deprecated, CNNE is experimental
#


description = "Train a deep learning model for nephropathy detection from ct scan of patients.\nTrain data is provided by https://doi.org/10.1016/j.dib.2025.111446, see README.md for details."

# handle command line args for choosing model, datafolder, hyperparameters, etc.
parser = argparse.ArgumentParser(description=description)
parser.add_argument("--train", choices=["cnn", "vgg", "cnne"], default="none", help="choose which model to train")
parser.add_argument("--data", default="traindata")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
parser.add_argument("--tanh", action="store_true", help="enable tanh in CNNE (disabled for now)")
parser.add_argument("--sigmoid", action="store_true", help="enable sigmoid in CNNE (disabled for now)")
parser.add_argument("--batch_size", type=int, default=64, help="batch size for DataLoader")
parser.add_argument("--train_split", type=float, default=0.8, help="fraction of data used for training (0-1)")
parser.add_argument("--load", action="store_true", help="load model from file (disabled for now)")
parser.add_argument("--finetune", choices=["efficientnet", "mobilenet"], default="none", help="use pretrained weights (if available)")
args = parser.parse_args()

# validate args
if args.batch_size < 1:
    parser.error("--batch_size must be >= 1")
if not 0.0 < args.train_split < 1.0:
    parser.error("--train_split must be between 0 and 1")
if args.train != "none" and args.finetune != "none":
    parser.error("cannot use --train and --finetune together")


channels = 3 if args.finetune != "none" else 1
# MyData handles loading/preprocessing/splitting
mydata = MyData(args.data, batch_size=args.batch_size, train_split=args.train_split, channels=channels)

if args.finetune == "efficientnet":
    print("INFO: using pretrained weights where available")
    mytrainer = KidneyEfficientNet(mydata, fine_tune="all")
    mytrainer.train(num_epochs=args.epochs)
    
    # results = mytrainer.analyze_validation_errors()
    # mytrainer.save_error_report(results, 'errori_validazione.txt')
elif args.finetune == "mobilenet":
    print("INFO: using pretrained weights where available")
    mytrainer = KidneyMobileNet(mydata, fine_tune=True)
    mytrainer.train(num_epochs=args.epochs)
    mytrainer.saveFeatureMaps()
elif args.finetune != "none":
    print("ERROR: finetuning is only available for efficientnet and mobilenet models")
    sys.exit(1)

if args.train == "cnn":
    mytrainer = KidneyCNNTrainer(mydata)
    mytrainer.train(num_epochs=args.epochs)
    mytrainer.saveModel()
    mytrainer.saveFeatureMaps()
elif args.train != "none":
    print("ERROR: unknown model type for training")
    sys.exit(1)

sys.exit(0)
