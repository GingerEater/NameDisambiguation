# encoding: utf-8


import file_config
import numpy as np
import xgboost as xgb
from sklearn import metrics

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

class XgboostRank(object):

    def __init__(self, model_path):
        self.model = None
        self.param = {'booster': 'gbtree', 'max_depth': 5, 'eta': 0.01, 'silent': 1, 'objective': 'rank:pairwise',
                      'gamma': 0.2, 'lambda': 700, 'subsample': 0.8, 'seed': 1}
        self.num_round = 1000

        self.model_path = model_path

    def read_data_from_file(self, file_name):
        """
        读取数据
        :param file_name:
        :return:
        """
        y_list = []
        x_list = []
        with open(file_name) as fp:
            for line in fp:
                uline = line.strip().decode('utf-8')
                features = uline.split('\t')[0]
                ulineList = features.split(' ')
                _y = np.int(ulineList[0])
                _x = [np.float(x) for x in ulineList[2:]]
                y_list.append(_y)
                x_list.append(_x)
        return np.array(y_list), np.array(x_list)

    def read_group(self, file_name):
        """
        读取rank数据对应的group文件
        :param file_name:
        :return:
        """
        group_list = []
        for line in open(file_name):
            uline = line.strip().decode('utf-8')
            group_count = np.int(uline)
            group_list.append(group_count)
        return np.array(group_list)

    def train_models(self, feature_path, group_path):
        """
        训练模型
        :param feature_path:
        :param group_path:
        :return:
        """
        y, x = self.read_data_from_file(feature_path)
        group_list = self.read_group(group_path)
        dtrain = xgb.DMatrix(x, label=y)
        dtrain.set_group(group_list)
        self.model = xgb.train(self.param, dtrain, self.num_round)
        self.model.save_model(self.model_path)
        self.model.dump_model(self.model_path + '.dump.txt')
        return

    def load_rank_model(self, model_path=None):
        """
        加载保存的模型
        :param model_path:
        :return:
        """
        _model_path = model_path if model_path else self.model_path
        self.model = xgb.Booster()
        self.model.load_model(_model_path)
        return self.model

    def compute_precision(self, y, preds, group_list):
        num = len(group_list)
        preci_list = []
        i = 0
        j = group_list[0]
        group_index = 0
        while 1:
            group_index += 1
            if group_index >= num:
                break
            _y_list = y[i:j]
            _preds_list = preds[i:j]

            y_dict = {}
            for i, val in enumerate(_y_list):
                y_dict[i] = val

            pred_dict = {}
            for i, val in enumerate(_preds_list):
                pred_dict[i] = val

            sort_y_list = sorted(y_dict.items(), key=lambda x: x[1])
            sort_pred_list = sorted(pred_dict.items(), key=lambda x: x[1])

            y_index_list = [ele[0] for ele in sort_y_list[:len(sort_y_list)/3]]
            pred_index_list = [ele[0] for ele in sort_pred_list[:len(sort_pred_list)/3]]

            correct_num = len(set(y_index_list) & set(pred_index_list))
            preci_list.append(float(correct_num) / (len(_y_list)/3))

            # else:
            #     self.badcase_file.write("label: " + str(i + _y_index) + "\tpredict: " + str(i + _preds_index) + "\n")

            i = j
            j = i + group_list[group_index]

        preci = sum(preci_list) / len(preci_list)
        return preci

    def predict_from_file(self, feature_path, group_path):
        """
        计算预测准确率
        :param feature_path:
        :param group_path:
        :return:
        """
        y, x = self.read_data_from_file(feature_path)
        dtest = xgb.DMatrix(x, label=y)
        group_list = self.read_group(group_path)
        dtest.set_group(group_list)
        preds = self.model.predict(dtest)

        # 计算group准确率
        precision = self.compute_precision(list(y), list(preds), list(group_list))
        print(precision)

    def cul_fea_weight(self):
        """
        计算每个feature的权重信息
        :return:
        """
        model = self.load_rank_model(self.model_path)
        importance = model.get_fscore()

        print(importance)

        # val_list = []
        # for fea, val in importance.items():
        #     val_list.append(val)
        #
        # fea_weight = [round(float(item)/sum(val_list), 3) for item in val_list]
        # print fea_weight


if __name__ == "__main__":
    xgboost = XgboostRank(file_config.xgboost_model_path)

    # xgboost.train_models(file_config.train_format_path, file_config.train_group_path)

    # xgboost.load_rank_model()
    # xgboost.xgboostpredict_from_file(file_config.train_format_path, file_config.train_group_path)

    xgboost.cul_fea_weight()
