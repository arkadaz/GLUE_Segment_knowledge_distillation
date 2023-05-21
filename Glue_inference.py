import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch.nn as nn
from PIL import Image
import PIL
import os


class Glue:
    def __init__(self, HALF: bool = False):
        self.HALF: bool = HALF
        self.DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_path: str = os.getcwd()
        self.PATH: str = "{}/U_Net_67_0.9401585877139755.pt".format(self.current_path)
        if self.HALF and self.DEVICE == "cuda":
            self.model_load: torch.jit._script.RecursiveScriptModule = self.load_model(
                path_to_load=self.PATH, DEVICE=self.DEVICE
            ).half()
        else:
            self.model_load: torch.jit._script.RecursiveScriptModule = self.load_model(
                path_to_load=self.PATH, DEVICE=self.DEVICE
            )

    def load_model(
        self, path_to_load: str = None, DEVICE="cpu"
    ) -> torch.jit._script.RecursiveScriptModule:
        assert type(path_to_load) is str, "path_to_load must be string type"
        assert type(DEVICE) is str, "DEVICE must be string type"

        model_load: torch.jit._script.RecursiveScriptModule = torch.jit.load(
            path_to_load, map_location=DEVICE
        )
        return model_load.eval()

    def release_memory(self):
        del self.model_load

    def transform_image(self, image: Image):
        image_infer_array: np.ndarray = np.array(image)
        IMAGE_HEIGHT = 256
        IMAGE_WIDTH = 256
        transform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.augmentations.transforms.Normalize(
                    mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
                ),
                ToTensorV2(),
            ]
        )
        return transform(image=image_infer_array)["image"].unsqueeze(0).to(self.DEVICE)

    def predict(self, image: Image):
        w, h = image.size
        image_to_predict = self.transform_image(image.convert("RGB"))
        with torch.no_grad():
            softmax = nn.Softmax(dim=1)
            preds = (
                torch.squeeze(
                    torch.argmax(softmax(image_to_predict), axis=1).permute(1, 2, 0)
                )
                .cpu()
                .detach()
                .numpy()
                * 255
            )
        return Image.fromarray(np.uint8(preds)).resize(
            size=(w, h), resample=PIL.Image.BILINEAR
        )
