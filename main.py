from flask import Flask, request, render_template

import gc

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image


def load_model():

    model_deploy = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    model_deploy.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    model_deploy.classifier = nn.Sequential(nn.Flatten(),
                                    nn.Linear(1280, 20),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(20, 1),
                                    nn.Sigmoid())

    for param in model_deploy.parameters():
        param.requires_grad = False

    model_state = torch.load('model_mobilenet.pt', map_location=torch.device('cpu'))
    model_deploy.load_state_dict(model_state)
    model_deploy.eval() # gets rid of batch norm, dropout

    gc.collect()

    return model_deploy


def convert_image_file_torch(image_file):
    return torch.tensor(np.asarray(Image.open(image_file).convert("RGB"))).permute(2, 0, 1)


def make_prediction(image_file):
    img = convert_image_file_torch(image_file)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    resize = transforms.Resize((224, 224))

    img = resize(img/255.)
    img = normalize(img)

    model_deploy = load_model()

    gc.collect()

    classes = model_deploy(img.float().unsqueeze(0)).numpy()
    prob_dog = classes[0][0]
    if prob_dog > 0.5:
        label = "dog"
        confidence =  np.round(100 * prob_dog, 2)
    else:
        label = "cat"
        confidence = np.round(100 * (1 - prob_dog), 2)

    return label, confidence


app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    """Basic HTML response."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return 'No file part in the request', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file.', 400

    pred, conf = make_prediction(file)
    return render_template("results.html", prediction = pred, confidence = conf)

if __name__ == '__main__':
    app.run(debug=True)
