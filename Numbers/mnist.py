import io
import json
import sys
import os
import re
import base64
import numpy as np
from PIL import Image
from model import SimpleNet
import torch
from flask import Flask, request
from flask_cors import CORS
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = '../models/'
MODEL_NAME = 'pytorch_computer_vision_model.pth'
MODEL_SAVE_PATH = MODEL_PATH + MODEL_NAME

model = SimpleNet(input_size=28*28, num_classes=10)

# Load in the saved state_dict()
model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Send model to GPU
model = model.to(device)

# Default output
res = {"result": 0,
       "data": [],
       "error": ''}

# Setup flask server
app = Flask(__name__)


# Setup url route which will calculate
# total sum of array.
@app.route("/mnist", methods=["POST"])
def sum_of_array():
    data = request.get_json()


    # Convert data url to numpy array
    img_str = re.search(r'base64,(.*)', data).group(1)
    image_bytes = io.BytesIO(base64.b64decode(img_str))
    im = Image.open(image_bytes)
    arr = np.array(im)[:, :, 0:1]

    # Normalize and invert pixel values
    arr = (255 - arr) / 255.

    # Predict class
    predictions = model(arr)

    # Return label data
    res['result'] = 1
    res['data'] = [float(num) for num in predictions]
    res.headers['Access-Control-Allow-Origin'] = '*'
    # Return data in json format
    return Flask.jsonify(res)


if __name__ == "__main__":
    app.run(debug=True)

