#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  :
# @Desc  :

import argparse
import os
import time

import cv2
import numpy as np
from lib.gcforest.gcforest import GCForest
from lib.gcforest.utils.config_utils import load_json


# sys.path.insert(0, "lib")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str, default='examples/demo_mnist-gc.json', help="gcfoest Net Model File")
    parser.add_argument("--prefix", dest="prefix", type=int, default=0)
    parser.add_argument("--data_num", dest="data_num", type=str, default='3')
    parser.add_argument("--save", dest="save", type=str, default='gc')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = load_json(args.model)
    print(args)

    total_start = time.process_time()
    prefix_path = '' #change the prefix_path in this
    
    # the path of flow data stores in this path
    data_path = prefix_path + '/train_test_data_argument/flow/flow' + args.data_num + '/'
    emotion = ['ne', 'po', 'sur']
    total_acc = 0

    test_tconf = None
    train_tconf = None
    rf_test_tconf = None
    rf_train_tconf = None
    start = time.process_time()
    
    # the gc results save in this path
    save_path = prefix_path + '/train_test_data_argument/apex_deepgc_' + args.save + '/' \
                + 'deepgc_' + args.save + args.data_num + '/'  

    print('data_path is {}'.format(data_path))
    print('save_path is {}'.format(save_path))

    if os.path.exists(save_path) is False:
        os.mkdir(save_path)

    for file in os.listdir(data_path):
        if file == 'total': continue
        train_data, test_data, train_label, test_label = [], [], [], []
        train_path = data_path + file + '/train/'
        for pic in os.listdir(train_path):
            if pic.split('_')[-1] == 'os.tif':
                img = cv2.imread(train_path + pic)
                img = cv2.resize(img, (28, 28))
                train_data.append(img)
                label = [i for i, each in enumerate(emotion) if each in pic]
                train_label.append(label[0])

        test_path = data_path + file + '/test/'
        for pic in os.listdir(test_path):
            if pic.split('_')[-1] == 'os.tif':
                img = cv2.imread(test_path + pic)
                img = cv2.resize(img, (28, 28))
                test_data.append(img)
                label = [i for i, each in enumerate(emotion) if each in pic]
                test_label.append(label[0])

        # -------------------------------------------
        train_data, test_data = np.array(train_data), np.array(test_data)
        train_label, test_label = np.array(train_label), np.array(test_label)

        print('train_data.shape is {}'.format(train_data.shape))
        print('test_data.shape is {}'.format(test_data.shape))

        X_train = train_data[:, np.newaxis, :, :]
        X_test = test_data[:, np.newaxis, :, :]

        model = GCForest(config)
        X_train_enc = model.fit_transform(X_train, train_label)

        try:
            if model.fg is not None:
                X_train_gc = model.gf_transfrom(X_train)
                X_test_gc = model.gf_transfrom(X_test)

                total_train_gc = None
                total_test_gc = None
                for inx, each in enumerate(X_train_gc):
                    each_gc = np.array(each).reshape(each.shape[0], -1)
                    if total_train_gc is None:
                        total_train_gc = each_gc
                    else:
                        total_train_gc = np.hstack((total_train_gc, each_gc))

                for inx, each in enumerate(X_test_gc):
                    each_gc = np.array(each).reshape(each.shape[0], -1)
                    if total_test_gc is None:
                        total_test_gc = each_gc
                    else:
                        total_test_gc = np.hstack((total_test_gc, each_gc))

                file_path = save_path + file + '/'
                if os.path.exists(file_path) is False:
                    os.mkdir(file_path)

                np.save(file_path + 'X_train_gc', total_train_gc)
                np.save(file_path + 'X_test_gc', total_test_gc)
                print('file_path {}, shape {}'.format(file_path, total_train_gc.shape))

        except Exception as e:
            X_train_gc = model.gf_transfrom(X_train)
            print('len transformation is {}'.format(len(X_train_gc)))
            print(e)
            exit()

        X_train_enc, x_train_layer_enc = model.transform(X_train)
        X_test_enc, x_test_layer_enc = model.transform(X_test)

        X_train_enc = X_train_enc.reshape((X_train_enc.shape[0], -1))
        X_test_enc = X_test_enc.reshape((X_test_enc.shape[0], -1))

        file_path = save_path + file + '/'
        if os.path.exists(file_path) is False:
            os.mkdir(file_path)

        np.save(file_path + 'X_train_enc', X_train_enc)
        np.save(file_path + 'X_test_enc', X_test_enc)
        print('file_path {}, shape {}'.format(file_path, X_train_enc.shape))
