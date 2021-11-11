# -*- coding: utf-8 -*-
# @File    : evaluation.py
# @Author  : smx
# @Date    : 2019/11/29 
# @Desc    : 
import numpy as np
import pandas as pd
import os


def evaluate(conf):
    UAR = np.mean([conf[i][i] / sum(conf[i]) for i in range(3)])
    UF1 = sum([2 * conf[i][i] / (sum(conf[:, i]) + sum(conf[i])) for i in range(3)]) / 3.0

    Acc = sum([conf[i][i] for i in range(3)]) / np.sum(conf)
    Recall = np.mean([conf[i][i] / sum(conf[i]) if sum(conf[i]) != 0 else 0 for i in range(3)])
    Precision = np.mean(
        [conf[i][i] / sum(conf[:, i]) if sum(conf[:, i]) != 0 else 0 for i in range(3)])
    F1 = 2 * Recall * Precision / sum([Recall, Precision])

    return round(Acc, 4), round(UAR, 4), round(UF1, 4), round(Recall, 4), round(Precision, 4), round(F1, 4)


def sava_matric(conf_list, name, prefix_path):
    conf_name = ['total', 'casme', 'smic', 'samm']
    conf_metric = list(map(evaluate, conf_list))
    metric_name = ['Acc', 'UAR', 'UF1', 'Recall', 'Precision', 'F1']
    acc_df = pd.DataFrame(data=[conf_metric[0], conf_metric[1], conf_metric[2], conf_metric[3]], columns=metric_name)
    print(acc_df)
    file_name = prefix_path + '/train_test_data_argument/compare_result/' + name + '_metrics.csv'
    acc_df.to_csv(file_name, header=True, index=True)
    print('metric save file location is {}'.format(file_name))

    emotion = ['ne', 'po', 'sur']
    conf_df = None
    for i in range(len(conf_name)):
        temp_df = pd.DataFrame(data=conf_list[i],
                               columns=[conf_name[i] + '_' + str(emotion[j]) for j in range(len(emotion))])
        if conf_df is None:
            conf_df = temp_df
        else:
            conf_df = pd.concat([conf_df, temp_df], axis=1)
    print(conf_df)
    save_path = prefix_path + '/train_test_data_argument/compare_result/'
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    file_name = save_path + name + '_conf.csv'
    conf_df.to_csv(file_name, header=True, index=True)
    print('confusion matrix save file location is {}'.format(file_name))


if __name__ == '__main__':
    conf = [[82, 5, 0], [12, 18, 2], [1, 0, 24]]
    res = evaluate(np.array(conf))
    print(res)
