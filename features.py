import os
import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class FeatureVisualizer(ABC):
    
    # here to force implementation in subclasses
    @abstractmethod
    def getModel(self):
        raise NotImplementedError
    
    @abstractmethod
    def getData(self):
        raise NotImplementedError

    def saveFeatureMaps(self, out_prefix="featmap", device=None):
        """
        model: any KidneyCNN compatible model having block1, block2, block3
        image_tensor: shape (1, 1, H, W) already normalized as in training
        saves 3 images: feature maps of block1, block2, block3 (grids)

        It saves every feature map (4D tensors) produced by each layer/module.
        This can become really large and messy with deep networks, like EfficientNet.
        """

        self.prefix_path = "Info_"+self.__class__.__name__+"/"
        if not os.path.exists(self.prefix_path):
            os.makedirs(self.prefix_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = self.getModel()
        image_tensor = self.getData().getImage(0)[0].unsqueeze(0)  # add batch dim

        model.eval()
        image_tensor = image_tensor.to(device)

        # capture intermediate activations with forward hooks
        activations = {}
        hooks = []

        def make_hook(name):
            def hook(module, inp, out):
                # store only 4D tensor activations (B, C, H, W)
                if isinstance(out, torch.Tensor) and out.dim() == 4:
                    activations[name] = out.detach().cpu()
            return hook

        for name, module in model.named_modules():
            # skip top-level module and containers
            if name == "":
                continue
            if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList, torch.nn.ModuleDict)):
                continue
            try:
                h = module.register_forward_hook(make_hook(name))
                hooks.append(h)
            except Exception:
                # some modules may not accept hooks; ignore
                pass

        with torch.no_grad():
            _ = model(image_tensor)

        # remove hooks
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

        if not activations:
            print("No 4D feature maps were captured from the model.")
            return

        # save each captured activation as a grid image
        for idx, (name, feat) in enumerate(activations.items()):
            # feat: (B, C, H, W)
            if not isinstance(feat, torch.Tensor) or feat.dim() != 4:
                continue
            B, C, H, W = feat.shape
            ncols = min(8, C) if C > 0 else 1
            safe_name = name.replace('.', '_')
            filename = f"{self.prefix_path}{out_prefix}_{idx}_{safe_name}.png"
            self.saveGrid(feat[0], filename, ncols=ncols)

        print(f"Feature maps saved to {self.prefix_path}")

    
    def saveGrid(self, feat_chw, filename, ncols=4):
        """
        feat_chw: tensor (C, H, W)
        normalizza ogni canale [0,1] e salva una griglia.
        """
        C, H, W = feat_chw.shape
        nrows = (C + ncols - 1) // ncols

        plt.figure(figsize=(ncols * 2.2, nrows * 2.2))
        for i in range(C):
            ax = plt.subplot(nrows, ncols, i + 1)
            fm = feat_chw[i].detach().cpu()

            # normalizzazione per visualizzare bene ogni canale
            fm = fm - fm.min()
            if fm.max() > 0:
                fm = fm / fm.max()

            ax.imshow(fm, cmap="gray")
            ax.axis("off")
            ax.set_title(f"ch {i}", fontsize=8)

        plt.tight_layout()
        plt.savefig(filename, dpi=200, bbox_inches="tight")
        plt.close()