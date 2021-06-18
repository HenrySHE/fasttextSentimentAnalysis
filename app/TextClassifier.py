# -*- coding: utf-8 -*-
"""
@Time ： 2020-12-02 14:41
@Auth ： liangpw3
@Description：

"""

from algo.Algo_interface import Algo_interface
import api as runs
import json
import numpy as np
import fasttext
import os
import pandas as pd

class TextClassifier(Algo_interface):
    def __init__(self, model_type, model_name, model_params):
        self.task_type = model_type
        self.model_name = model_name
        self.model_params = model_params
        self.model = None
        self.build_model()
        # return self.model

    def set_model(self, model):
        self.model = model
        return 1

    def get_model(self):
        return self.model

    def build_model(self):
        if self.model_name == 'fasttext':      #  由于fb-fasttext并没有定义模型类这一步，是直接训练的，直接pass
            # self.model = linear_model.LogisticRegression(**self.model_params)
            pass

    def train(self, data):
        x_train, y_train = data   # 读取Series
        def fasttext_preprocess(AfterFilterSenS, data_label):
            temp_df = pd.DataFrame()
            temp_df['cut_sentence'] = AfterFilterSenS
            temp_df['label'] = data_label
            temp_df['fasttext_training_data'] = "__label__" + temp_df['label'] + " , " + temp_df['cut_sentence']
            return temp_df['fasttext_training_data'].tolist()
        #预处理为fb-fasttext专用格式
        fasttext_sentences_train = fasttext_preprocess(x_train, y_train)
        #生成fb-fasttext临时文件
        with open('./temp_train_data.txt', 'w', encoding='utf-8') as temp:
            for sentence in fasttext_sentences_train:
                temp.write(sentence + "\n")
        # print("fb-fasttext done!")
        # 训练获取模型，fb-fasttext模型是直接用方法训练来获取的
        # 参数后面改外部传入**self.model_params

        dim,epoch,lr,wordNgrams,loss = self.model_params['dim'],self.model_params['epoch'],self.model_params['lr'],self.model_params['wordNgrams'],self.model_params['loss']
        self.model = fasttext.train_supervised(input='./temp_train_data.txt', dim=dim, epoch=epoch,
                                               lr=lr, wordNgrams=wordNgrams, loss=loss)
        try:
            os.remove('./temp_train_data.txt')  #清除临时文件
        except Exception as e:
            pass
        return 1

    def predict(self, data):
        # data进来是Series，对接的api.py那里的
        y_pred = []
        for i in data.index:
            y_pred.append(self.model.predict(data[i])[0][0][9:])
        y_pred = np.asarray(y_pred)
        return y_pred