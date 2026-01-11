import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage
from torchvision import transforms
from torch.utils.data import random_split

from collections import defaultdict
from torch.utils.data import Subset

import os, csv, re, random

class Data:
    def __init__(self, datafolder="traindata", batch_size=64, train_split=0.8, channels=1):
        self.setTrainLoader(None)
        self.setValLoader(None)

        self.channels = channels
        print(f"INFO: Data: using {self.channels} channel(s) for images")

        if not datafolder:
            print("ERROR: Data: didn't set training data correctly")
        
        if not os.path.isdir(datafolder):
            print("ERROR: Data: folder to locate data in seems not to be a correct folder (missing or wrong type)")
        
        self.datafolder = datafolder

        # generate preprocessed data if not already present
        # this may break info contained in filename (patient, subsequent scans...)
        # TODO: fix ^^^

        preprocessed_path = "preprocessed_" + self.datafolder

        if not os.path.isdir(preprocessed_path):
            print("INFO: preprocessing and saving data to preprocessed folder")
            self.preprocess(savepath=preprocessed_path, overwrite=False)
        
        print("INFO: loading preprocessed data")
        self.load_preprocessed(savepath=preprocessed_path, batch_size=batch_size, train_split=train_split)
        
    def preprocess(self, savepath="", overwrite=False):
        """
        Load and save preprocessed dataset to disk in a folder-structure compatible with torchvision.datasets.ImageFolder,
        one subfolder per class. Also writes mapping.csv with (saved_fname, original_path, class_idx, class_name).

        If overwrite is False and savepath exists and is not empty, abort.
        """

        if savepath == "":
            savepath = "preprocessed_" + self.datafolder

        cnnNormalization = transforms.Normalize(mean=[0.5], std=[0.5])
        efficientnetNormalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=self.channels),   # force 1 or 3 channels
            transforms.Resize((224, 224)),                 # resize
            transforms.ToTensor(),                         # [0, 255] -> [0, 1]
            cnnNormalization if self.channels == 1 else efficientnetNormalization # apply correct normalization
        ])

        full_dataset = datasets.ImageFolder(root=self.datafolder, transform=transform)

        if os.path.isdir(savepath):
            # check directory empty or overwrite permitted
            if not overwrite and any(os.scandir(savepath)):
                print("ERROR: ", savepath, " folder already exists and is not empty. Use overwrite=True to replace.")
                return
        else:
            os.makedirs(savepath, exist_ok=True)

        to_pil = ToPILImage()
        mapping = []
        # full_dataset is ImageFolder with transform applied; samples contains original paths and class_idx
        for idx, (img_tensor, label) in enumerate(full_dataset):
            class_name = full_dataset.classes[label]
            class_dir = os.path.join(savepath, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Denormalize from ~[-1,1] back to [0,1] then to PIL
            img = img_tensor.clone()
            img = img * 0.5 + 0.5
            img = torch.clamp(img, 0.0, 1.0)

            # Build filename
            # fname = f"{class_name}_{idx:06d}.png"
            # outpath = os.path.join(class_dir, fname)

            # nome originale
            original_path = full_dataset.samples[idx][0]
            original_fname = os.path.basename(original_path)

            outpath = os.path.join(class_dir, original_fname)

            # Convert to PIL and save
            pil_img = to_pil(img)
            pil_img.save(outpath)

            # original path if available
            original_path = None
            try:
                original_path = full_dataset.samples[idx][0]
            except Exception:
                original_path = ""

            mapping.append((os.path.join(class_name, original_fname), original_path, label, class_name))

        # write mapping.csv
        csvpath = os.path.join(savepath, "mapping.csv")
        with open(csvpath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["saved_fname", "original_path", "class_idx", "class_name"])
            writer.writerows(mapping)

        print(f"Saved {len(mapping)} images into '{savepath}' with mapping file '{csvpath}'")

    def get_patient_id(self, path):
        PATIENT_RE = re.compile(r"^(P\d{3,})_", re.IGNORECASE)  # P037_...

        fname = os.path.basename(path)
        m = PATIENT_RE.match(fname)
        if not m:
            raise ValueError(f"Nome file non valido (atteso 'P###_...'): {fname}")
        return m.group(1).upper()
    
    def split_by_patient(self, imagefolder_ds, train_split=0.8, seed=42):
        # 1) group patients by index and verify label per patient
        patient_to_indices = defaultdict(list)
        patient_to_label = {}

        for idx, (path, y) in enumerate(imagefolder_ds.samples):
            pid = self.get_patient_id(path)
            patient_to_indices[pid].append(idx)
            if pid in patient_to_label and patient_to_label[pid] != y:
                raise ValueError(
                    f"Paziente {pid} ha immagini in più classi! "
                    f"{patient_to_label[pid]} vs {y}. Controlla il dataset."
                )
            patient_to_label[pid] = y

        # 2) list patients per class
        label_to_patients = defaultdict(list)
        for pid, y in patient_to_label.items():
            label_to_patients[y].append(pid)

        rng = random.Random(seed)

        train_pids, val_pids = set(), set()

        for y, pids in label_to_patients.items():
            rng.shuffle(pids)
            n_train = int(train_split * len(pids))
            train_pids.update(pids[:n_train])
            val_pids.update(pids[n_train:])

        # 3) build indexes
        train_indices = [i for pid in train_pids for i in patient_to_indices[pid]]
        val_indices   = [i for pid in val_pids   for i in patient_to_indices[pid]]

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)

        train_ds = Subset(imagefolder_ds, train_indices)
        val_ds   = Subset(imagefolder_ds, val_indices)

        return train_ds, val_ds, train_pids, val_pids    

    def load_preprocessed(self, savepath="", batch_size=64, train_split=0.8):
        """
        Reloads the saved preprocessed folder into train/val DataLoaders.
        Assumes images in savepath are standard image files (PNG/JPG) already resized/grayscaled.
        Applies ToTensor + Normalize([0.5],[0.5]) to map [0,1] -> [-1,1] like original preprocess.
        """
        if savepath == "":
            savepath = "preprocessed_" + self.datafolder

        if not os.path.isdir(savepath):
            print("ERROR: load_preprocessed: savepath not found")
            return None, None

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=self.channels),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.dataset = datasets.ImageFolder(root=savepath, transform=transform)
        total = len(self.dataset)
        train_size = int(train_split * total)
        val_size = total - train_size
        # self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

        self.train_dataset, self.val_dataset, train_pids, val_pids = self.split_by_patient(
            self.dataset, train_split=train_split, seed=42
        )

        print("Overlap pazienti:", len(train_pids & val_pids))  # must be 0
        print("Train images:", len(self.train_dataset), "Val images:", len(self.val_dataset))

        self.setTrainLoader(DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True))
        self.setValLoader(DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False))

    def setTrainLoader(self, tl):
        self.train_loader = tl
    
    def setValLoader(self, vl):
        self.val_loader = vl

    def getTrainLoader(self):
        if self.train_loader == None:
            print("ERROR: MyData.getTrainLoader() is None, probably MyData was not initialized correctly")
        else:
            return self.train_loader
    
    def getValLoader(self):
        if self.val_loader == None:
            print("ERROR: MyData.getTrainLoader() is None, probably MyData was not initialized correctly")
        else:
            return self.val_loader
        
    def getImage(self, index):
        """
        returns (image_tensor, label) at given index from the full dataset
        """
        if not hasattr(self, 'dataset'):
            print("ERROR: MyData.getImage(): dataset not loaded")
            return None, None
        if index < 0 or index >= len(self.dataset):
            print("ERROR: MyData.getImage(): index out of range")
            return None, None
        return self.dataset[index]
            
