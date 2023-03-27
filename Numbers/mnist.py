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
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
from flask import Flask, request
from flask_cors import CORS
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = '../models/'
MODEL_NAME = 'pytorch_computer_vision_model.pth'
MODEL_SAVE_PATH = MODEL_PATH + MODEL_NAME

model = SimpleNet(input_size=28*28, num_classes=10)

model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

model = model.to(device)


def predict(data):

    res = {"result": 0,
           "data": [],
           "error": ''}

    img_str = re.search(r'base64,(.*)', data).group(1)
    image_bytes = io.BytesIO(base64.b64decode(img_str))
    im = Image.open(image_bytes)
    arr = np.array(im)[:, :, 0:1]

    arr = (255 - arr) / 255.

    predictions = model(arr)

    res['result'] = 1
    res['data'] = [float(num) for num in predictions]
    res.headers['Access-Control-Allow-Origin'] = '*'
    # Return data in json format
    return json.dumps(res)


class Server(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        f = open("index.html")

        self.wfile.write(bytes(f, 'utf-8'))



    def do_POST(self):
        if self.path == "/send-img":
            data = self.rfile.read(int(self.headers.get('Content-Length'))).decode("utf-8")
            return predict(data)


if __name__ == "__main__":
    webServer = HTTPServer(("localhost", 5000), Server)
    print("Server started http://%s:%s" % ("localhost", 5000))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")

