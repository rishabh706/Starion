from flask import Flask, jsonify, request
from model import WM
from datetime import datetime
import base64
import json
import cv2
import numpy as np
import os

app = Flask(__name__)


model = WM()
model.load_network()


@app.route("/model/predict", methods=["GET", "POST"])
def predict():

    data = json.loads(request.data)
    img = data.get("imageBase64")
    imageFile = data.get("imageFileName")

    print(imageFile)
    cords = model.get_predictions(imageFile)
    cords["status"] = "200"
    return jsonify(cords)


##    image=base64.b64decode(str(img))
##    decoded_img = cv2.imdecode(np.frombuffer(image, np.uint8),-1)
##
##    result=model.run_ocr(decoded_img[:,:,1:4],vis=False)
##
##
##    return jsonify(result)
##


@app.route("/server", methods=["GET", "POST"])
def server():
    return "running"


if __name__ == "__main__":
    app.run("192.168.0.115", port=5000, use_reloader=False, debug=False, threaded=True)
