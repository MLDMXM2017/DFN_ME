#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc  : DFN model

import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from sklearn import metrics

from Simple_CNN import Simple_CNN
from evaluation import evaluate

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def get_data(file, data_path, apex_data_path, gc_data_path, flow_train_data, flow_test_data, apex_train_data,
             apex_test_data,
             train_label, test_label):
    # the input of 1st channel
    train_path = data_path + file + '/train/'
    for pic in os.listdir(train_path):
        if pic.split('_')[-1] == 'os.tif':
            img = cv2.imread(train_path + pic)
            img = cv2.resize(img, (img_size, img_size))
            flow_train_data.append(img)
            label = [i for i, each in enumerate(emotion) if each in pic]
            train_label.append(label[0])
    if len(flow_test_data) == 0:
        test_path = data_path + file + '/test/'
        for pic in os.listdir(test_path):
            if pic.split('_')[-1] == 'os.tif':
                img = cv2.imread(test_path + pic)
                img = cv2.resize(img, (img_size, img_size))
                flow_test_data.append(img)
                label = [i for i, each in enumerate(emotion) if each in pic]
                test_label.append(label[0])

    # the input of 2rd channel
    train_path = apex_data_path + file + '/train/'
    for pic in os.listdir(train_path):
        img = cv2.imread(train_path + pic)
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        apex_train_data.append(img)

    if len(apex_test_data) == 0:
        test_path = apex_data_path + file + '/test/'
        for pic in os.listdir(test_path):
            img = cv2.imread(test_path + pic)
            img = cv2.resize(img, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            apex_test_data.append(img)

    # the input of 3th channel
    file_path = gc_data_path + file + '/'
    enc_train_data = np.load(file_path + 'X_train_' + in3_name + '.npy')
    enc_test_data = np.load(file_path + 'X_test_' + in3_name + '.npy')
    return enc_train_data, enc_test_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in3file", dest='in3file', default='flow_deepgc_gc', help="deepgc folder location", type=str)
    parser.add_argument("--in3name", dest='in3name', default='gc', help="deepgc file name", type=str)
    parser.add_argument("--model", dest='model', default='Simple_CNN', help="model name", type=str)
    parser.add_argument("--resampling", dest='resampling', default='false', help="resample method", type=str)
    parser.add_argument("--batch_size", dest='batch_size', default='32', help="batch size", type=str)
    parser.add_argument("--learning_rate", dest='learning_rate', default='0.00001', help="learning_rate", type=str)

    args = parser.parse_args()

    print(args)

    in3_file = args.in3file
    in3_name = args.in3name

    img_size = 28
    total_start = time.process_time()
    prefix_path = ''   #you can change your path in this
    data_path = prefix_path + '/train_test_data_argument/flow/flow1/'      #the path of flow data 
    apex_data_path = prefix_path + '/train_test_data_argument/raw_002/raw1/'  # the path of apex data
    gc_data_path = prefix_path + '/train_test_data_argument/' + in3_file + '/' + in3_file  # the path of gc results
    sample_path = prefix_path + '/train_test_data_argument/resampling/' + args.resampling + '/'

    emotion = ['ne', 'po', 'sur']
    conf_name = ['total', 'casme', 'smic', 'samm']
    conf_list = [None for i in range(4)]

    start = time.process_time()
    # LOSO
    for file in os.listdir(data_path):
        if file == 'total': continue

        flow_train_data, flow_test_data, train_label, test_label = [], [], [], []
        apex_train_data, apex_test_data = [], []
        enc_train_data, enc_test_data = None, None

        for iter_num in range(0, 3):
            if iter_num == 2: iter_num = 5  # The last way to ensure enhancement is to flip vertically
            temp_data_path = data_path[:-2] + str(iter_num) + '/'
            temp_apex_path = apex_data_path[:-2] + str(iter_num) + '/'
            temp_gc_path = gc_data_path + str(iter_num) + '/'
            print('temp_data_path is {}'.format(temp_data_path))
            print('temp_apex_path is {}'.format(temp_apex_path))
            print('temp_gc_path is {}'.format(temp_gc_path))
            enc_tr_data, enc_tt_data = get_data(file, temp_data_path, temp_apex_path, temp_gc_path,
                                                flow_train_data, flow_test_data, apex_train_data, apex_test_data,
                                                train_label, test_label)
            if enc_train_data is not None:
                enc_train_data = np.vstack((enc_train_data, enc_tr_data))
            else:
                enc_train_data, enc_test_data = enc_tr_data, enc_tt_data

        enc_train_data = np.array(enc_train_data).reshape((-1, enc_train_data.shape[1], 1))
        enc_test_data = np.array(enc_test_data).reshape((-1, enc_test_data.shape[1], 1))
        print('np.shape(np.array(enc_train_data)) {}'.format(np.shape(np.array(enc_train_data))))
        print('np.shape(np.array(enc_test_data)) {}'.format(np.shape(np.array(enc_test_data))))

        flow_train_data = np.array(flow_train_data).reshape((-1, img_size, img_size, 3))
        flow_test_data = np.array(flow_test_data).reshape((-1, img_size, img_size, 3))
        print('np.shape(np.array(flow_train_data)) {}'.format(np.shape(np.array(flow_train_data))))
        print('np.shape(np.array(flow_test_data)) {}'.format(np.shape(np.array(flow_test_data))))

        apex_train_data = np.array(apex_train_data).reshape((-1, img_size, img_size, 1))
        apex_test_data = np.array(apex_test_data).reshape((-1, img_size, img_size, 1))
        print('np.shape(np.array(apex_train_data)) {}'.format(np.shape(np.array(apex_train_data))))
        print('np.shape(np.array(apex_test_data)) {}'.format(np.shape(np.array(apex_test_data))))

        # ============================================================================

        model = Simple_CNN(inshape=(img_size, img_size, 3), class_num=len(set(train_label)),
                           batch_size=int(args.batch_size),
                           learning_rate=float(args.learning_rate), epochs=500)
        model.train(flow_train_data, apex_train_data, enc_train_data, train_label)
        pred = model.predict(flow_test_data, apex_test_data, enc_test_data)

        # ============================================================================
        # merge metrix
        overall_accuracy = metrics.accuracy_score(test_label, pred)
        temp_conf = metrics.confusion_matrix(test_label, pred, labels=[0, 1, 2])
        print('test {} person, acc is {}'.format(file, overall_accuracy))
        print('test confusion_matrix\n {}'.format(temp_conf))

        conf_list[0] = np.array(temp_conf) if conf_list[0] is None else conf_list[0] + np.array(temp_conf)
        if 'sub' in file:
            inx = 1
        else:
            inx = 2 if file[0] == 's' else 3
        conf_list[inx] = np.array(temp_conf) if conf_list[inx] is None else conf_list[inx] + np.array(temp_conf)

    # ms
    cost_time = round((time.time() - total_start) / 36000000, 0)
    conf_metric = list(map(evaluate, conf_list))
    metric_name = ['Acc', 'UAR', 'UF1', 'Recall', 'Precision', 'F1']
    df = pd.DataFrame(data=[conf_metric[0], conf_metric[1], conf_metric[2], conf_metric[3], [cost_time] * 6],
                      columns=metric_name, index=conf_name + ['time'])
    print('--------------------------------------------------------------------------------')
    print(df)

    file_name = os.getcwd() + '/train_test_data_argument/compare_result/chang_params/' + args.model + '_metrics.csv'
    df.to_csv(file_name, header=True, index=True)
    print('save file location is {}'.format(file_name))

    print('--------------------------------------------------------------------------------')
    emotion = ['ne', 'po', 'sur']
    conf_df = None
    for i in range(len(conf_name)):
        temp_df = pd.DataFrame(data=conf_list[i],
                               columns=[conf_name[i] + '_' + str(emotion[j]) for j in range(len(emotion))])
        if conf_df is None:
            conf_df = temp_df
        else:
            conf_df = pd.concat([conf_df, temp_df], axis=1)
    file_name = os.getcwd() + '/train_test_data_argument/compare_result/chang_params/' + args.model + '_conf.csv'
    conf_df.to_csv(file_name, header=True, index=True)
    print('--------------------------------------------------------------------------------')
    print(conf_df)
    print('save file location is {}'.format(file_name))
