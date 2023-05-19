from Glue_inference import Glue
import os
from PIL import Image
import torch


def test_predict():
    # load image
    current_dir = os.getcwd()
    image_path = current_dir + "/" + "image_test"
    image_names = os.listdir(image_path)
    image_full_path = []
    for image_name in image_names:
        if "_1.jpg" in image_name:
            image_full_path.append(image_path + "/" + image_name)

    # load model
    model = Glue(HALF=False)

    # predict
    for i in image_full_path:
        image_infer = Image.open(i)
        w, h = image_infer.size
        result = model.predict(image_infer)
        assert result.size == (w, h)


def test_cuda():
    assert "cuda" if torch.cuda.is_available() else "cpu" == "cuda"
