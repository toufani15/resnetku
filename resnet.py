# %matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
from keras import utils
from keras.models import Model
from keras.layers import *
from keras.datasets import cifar10

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

tf.test.gpu_device_name()

# def set_gpu_mem_alloc(mem_use):
#     avail  = 4041
#     percent = mem_use / avail
#     config = tf.ConfigProto()
#     config.gpu_options.per_process_gpu_memory_fraction = percent
#     config.gpu_options.visible_device_list = "0"
#     set_session(tf.Session(config=config))
    
# set_gpu_mem_alloc(3000)

# load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[:20000]
y_train = y_train[:20000]

x_test = x_test[:5000]
y_test = y_test[:5000]

# cari tau datanya, keadaan nya seperti apa
print(type(x_train))
print(x_train.shape)

print(type(y_train))
print(y_train.shape)

idx = 3
plt.imshow(x_train[idx])
print(y_train[idx])

# normalisasi dataset
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

num_class = 10
y_train = utils.np_utils.to_categorical(y_train, num_class)
y_test = utils.np_utils.to_categorical(y_test, num_class)

# arsitektur
def residual_block(inputz, filterz):
    bc1 = Conv2D(filterz, kernel_size=(3,3), padding='same')(inputz)
    bbn1 = BatchNormalization()(bc1)
    bac1 = ReLU()(bbn1)
#     dr1 = Dropout(0.25)(bac1)
    bc2 = Conv2D(filterz, kernel_size=(3,3), padding='same')(bac1)
    bbn2 = BatchNormalization()(bc2)
    add1 = Concatenate()([inputz, bbn2])
    bac2 = ReLU()(add1)
#     dr1 = Dropout(0.25)(bac1)
    return bac2

# ishape = x_train[0].shape
ishape = (32, 32, 3)

inp = Input(shape=ishape)
# 32x32
c1 = Conv2D(filters=32, kernel_size=(7,7), strides=(2,2), padding='same')(inp)
# 16x16

p1 = MaxPooling2D(pool_size=(2, 2))(c1)
# 8x8
r1 = residual_block(inputz=p1, filterz=32)
r11 = residual_block(inputz=r1, filterz=32)
r12 = residual_block(inputz=r11, filterz=32)

# 8x8

p2 = MaxPooling2D(pool_size=(2, 2))(r12)
# 4x4
r2 = residual_block(inputz=p2, filterz=64)
r21 = residual_block(inputz=r2, filterz=64)
r22 = residual_block(inputz=r21, filterz=64)

# 4x4

p4 = AveragePooling2D(pool_size=(2,2))(r22)
fl1 = Flatten()(p4)
fc = Dense(num_class, activation='softmax')(fl1)
output = fc

model = Model(inp, output)
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
model.summary()
# residual block


model.fit(
    x_train, y_train,
    validation_data=(x_test,y_test),
    epochs=10, batch_size=64
)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

