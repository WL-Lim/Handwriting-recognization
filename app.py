import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import cv2
import tensorflow as tf
from tensorflow import keras
import re
import base64

app = Flask(__name__)

def convert_img(imgdata):
    imgstr = re.search(r"base64,(.*)", str(imgdata)).group(1)
    with open("output.png", "wb") as output:
        output.write(base64.b64decode(imgstr))
        
oh_tfr = joblib.load("one_hot_transformer.pkl")
lb_map = {
    0: "0", 
    1: "1", 
    2: "2", 
    3: "3", 
    4: "4", 
    5: "5", 
    6: "6", 
    7: "7", 
    8: "8", 
    9: "9", 
    10: "A", 
    11: "B", 
    12: "C", 
    13: "D", 
    14: "E", 
    15: "F", 
    16: "G", 
    17: "H", 
    18: "I", 
    19: "J", 
    20: "K", 
    21: "L", 
    22: "M", 
    23: "N", 
    24: "O", 
    25: "P", 
    26: "Q", 
    27: "R", 
    28: "S", 
    29: "T", 
    30: "U", 
    31: "V", 
    32: "W", 
    33: "X", 
    34: "Y", 
    35: "Z", 
    36: "a", 
    37: "b", 
    38: "d", 
    39: "e", 
    40: "f", 
    41: "g", 
    42: "h", 
    43: "n", 
    44: "q", 
    45: "r", 
    46: "t", 
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict/", methods = ["GET", "POST"])
def predict():
    imgdata = request.get_data()
    convert_img(imgdata)
    x = cv2.imread("output.png")
    x = cv2.resize(x, (28, 28))
    x = np.mean(x, axis = 2) / 255
    x = x.reshape((1, 28 , 28 , 1))
    model = keras.models.load_model("emnist_balanced.h5")
    y = model.predict(x)
    label = oh_tfr.inverse_transform(y).reshape([len(y), ])
    label = pd.Series(label).map(lb_map)[0]
    return str(label)

if __name__ == "__main__":
    app.run()

