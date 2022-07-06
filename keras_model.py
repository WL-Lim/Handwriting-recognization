import os
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers

from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from keras import layers

train_path = r"C:\Users\Applg\Downloads\emnist_balanced\emnist-balanced-train.csv"
test_path = r"C:\Users\Applg\Downloads\emnist_balanced\emnist-balanced-test.csv"

df_train = pd.read_csv(train_path, header = None)
df_test = pd.read_csv(test_path, header = None)

X_train_full, y_train_full = (df_train.iloc[:, 1:] / 255), (df_train.iloc[:, :1])
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = \
  train_test_split(X_train_full, y_train_full, train_size = 0.8, shuffle = True, random_state = 42)

X_test, y_test = (df_test.iloc[:, 1:] / 255), (df_test.iloc[:, :1])

X_train_np = X_train.values.reshape([len(X_train), 28, 28, 1]).transpose([0, 2, 1, 3])
X_valid_np = X_valid.values.reshape([len(X_valid), 28, 28, 1]).transpose([0, 2, 1, 3])
X_test_np = X_test.values.reshape([len(X_test), 28, 28, 1]).transpose([0, 2, 1, 3])

from sklearn.preprocessing import OneHotEncoder
one_hot = OneHotEncoder()
y_train_oh = one_hot.fit_transform(y_train.values).toarray()
y_valid_oh = one_hot.transform(y_valid.values).toarray()
y_test_oh = one_hot.transform(y_test.values).toarray()

import joblib
joblib.dump(one_hot, r"C:\Users\Applg\OneDrive\桌面\新增資料夾\Data_Science\practice\my_deploy\one_hot_transformer.pkl")

# y_train_re = y_train.values.reshape(len(y_train), )
# for n in range(26, 46):
#   idx = np.where(y_train_re == n)
#   for i in range(3):
#     plt.imshow(X_train_np[idx[0][i], :, :, 0])
#     plt.title(y_train_re[idx[0][i]])
#     plt.show()

# y_valid_re = y_valid.values.reshape(len(y_valid), )
# for n in range(len(np.unique(y_valid_re))):
#   idx = np.where(y_valid_re == n)
#   i = 0
#   plt.imshow(X_valid_np[idx[0][i], :, :, 0])
#   plt.title(y_valid_re[idx[0][i]])
#   plt.show()

from functools import partial
CustomConv2D = partial(
        layers.Conv2D, 
        padding = "same", 
        activation = "elu", 
        use_bias = False, 
    )

model = keras.models.Sequential([
        layers.Input(shape = [28, 28, 1]), 
        layers.BatchNormalization(), 
        CustomConv2D(filters = 64, kernel_size = 3), 
        layers.MaxPool2D(2), 
        layers.BatchNormalization(), 
        CustomConv2D(filters = 128, kernel_size = 3), 
        layers.MaxPool2D(2), 
        layers.BatchNormalization(),
        # CustomConv2D(filters = 256, kernel_size = 3), 
        # layers.BatchNormalization(),
        # CustomConv2D(filters = 512, kernel_size = 3), 
        # layers.BatchNormalization(),
        # CustomConv2D(filters = 512, kernel_size = 3), 
        # layers.BatchNormalization(),
        # CustomConv2D(filters = 256, kernel_size = 3), 
        # layers.BatchNormalization(),
        CustomConv2D(filters = 128, kernel_size = 3),
        layers.MaxPool2D(2),  
        layers.BatchNormalization(),
        CustomConv2D(filters = 64, kernel_size = 3), 
        layers.MaxPool2D(2), 
        layers.BatchNormalization(), 
        CustomConv2D(filters = 47, kernel_size = 1, padding = "valid", activation = "softmax"), 
        layers.Flatten(), 
    ])
model.compile(loss = "categorical_crossentropy", 
       optimizer = "adam", 
       metrics = ["accuracy"])

model.summary()

epochs = 10
checkpoint_cb = keras.callbacks.ModelCheckpoint("emnist_balanced.h5")
history = model.fit(X_train_np, y_train_oh, epochs = epochs, validation_data = (X_valid_np, y_valid_oh), callbacks = [checkpoint_cb])

pd.DataFrame(history.history).plot(figsize = (10, 8))
plt.xlim([0, epochs - 1])
plt.grid(True)

model.evaluate(X_test_np, y_test_oh)