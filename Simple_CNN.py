# -*- coding: utf-8 -*-
# @File    : Simple_CNN_6614.py
# @Author  : smx
# @Date    : 2019/11/18 
# @Desc    : 现在的分数是71

import numpy as np
from keras import Input, Model
from keras.initializers import RandomNormal
from keras.layers import Conv1D
from keras.layers import Dense, Conv2D, MaxPooling2D, concatenate, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical

#from . import base_model


'''class Simple_CNN(base_model.Base_Model):
    def _build_graph(self):

        dim = 582

        input1 = Input(shape=(28, 28, 3), name='input1')
        input2 = Input(shape=(28, 28, 1), name='input2')
        input3 = Input(shape=(dim, 1), name='input3')

        initzer = RandomNormal(mean=0.0, stddev=0.01, seed=None)

        # first input
        x1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', kernel_initializer=initzer,
                    bias_initializer='zeros', padding='same', name='x1_conv1')(input1)
        x1 = MaxPooling2D(pool_size=(2, 2), strides=(3, 3), padding='same', name='x1_pool1')(x1)
        x1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', kernel_initializer=initzer,
                    bias_initializer='zeros', padding='same', name='x1_conv2')(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), strides=(3, 3), padding='same', name='x1_pool2')(x1)

        # second input
        x2 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', kernel_initializer=initzer,
                    bias_initializer='zeros', padding='same', name='x2_conv1')(input2)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(3, 3), padding='same', name='x2_pool1')(x2)
        x2 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', kernel_initializer=initzer,
                    bias_initializer='zeros', padding='same', name='x2_conv2')(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), strides=(3, 3), padding='same', name='x2_pool2')(x2)

        # ====================================
        # concat
        x = concatenate([x1, x2])
        x = Flatten()(x)

        # ====================================
        # third input
        x3 = Conv1D(16, kernel_size=(3), strides=(5), padding='same', activation='relu', kernel_initializer=initzer,
                    bias_initializer='zeros', name='x3_conv1')(input3)
        x3 = Flatten()(x3)

        # ====================================
        # concat
        x = concatenate([x, x3])
        x = Dense(512, activation='relu', name='dense1')(x)
        x = Dense(256, activation='relu', name='dense2')(x)
        output = Dense(self.class_num, activation='softmax', name='output')(x)

        model = Model(inputs=[input1, input2, input3], outputs=[output])
        Adam_optimizer = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=Adam_optimizer, metrics=['accuracy'])
        model.summary()
        return model

    def train(self, train_data1, train_data2, train_data3, train_label, valid_data=None, valid_label=None,
              layer_name=''):

        train_label = to_categorical(train_label, len(set(train_label)))
        if layer_name != '':
            self.model = Model(inputs=[self.model.get_layer('x1_conv1').input, self.model.get_layer('x2_conv1').input,
                                       self.model.get_layer('x3_conv1').input],
                               outputs=self.model.get_layer(layer_name).output)
            return

        if valid_data is not None:
            self.model.fit([train_data1, train_data2, train_data3], train_label,
                           validation_data=(valid_data, valid_label),
                           batch_size=self.batch_size, shuffle=True,
                           epochs=self.epochs, verbose=self.verbose, callbacks=self.callbacks)
        else:
            self.model.fit([train_data1, train_data2, train_data3], train_label, batch_size=self.batch_size,
                           epochs=self.epochs, shuffle=True,
                           verbose=self.verbose, callbacks=self.callbacks, validation_split=0.1)
        return

    def predict(self, test_data1, test_data2, test_data3, layer_name=''):
        if layer_name == '':
            pred = self.model.predict([test_data1, test_data2, test_data3], batch_size=self.batch_size)
            return np.argmax(pred, axis=1)
        else:
            pred = self.model.predict([test_data1, test_data2, test_data3], batch_size=self.batch_size)
            pred = np.reshape(pred, (len(test_data1), -1))
            return pred
'''

input1 = Input(shape=(170, 140, 3), name='input1')
input2 = Input(shape=(170, 140, 3), name='input2')

initzer = RandomNormal(mean=0.0, stddev=0.01, seed=None)

# first input
x1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', kernel_initializer=initzer,
            bias_initializer='zeros', padding='same', name='x1_conv1')(input1)
print(x1)
x1 = MaxPooling2D(pool_size=(2, 2), strides=(3, 3), padding='same', name='x1_pool1')(x1)
print(x1)
x1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', kernel_initializer=initzer,
            bias_initializer='zeros', padding='same', name='x1_conv2')(x1)
x1 = MaxPooling2D(pool_size=(2, 2), strides=(3, 3), padding='same', name='x1_pool2')(x1)


# ====================================
# concat
#x = concatenate([x1, x2])
#x = Flatten()(x)