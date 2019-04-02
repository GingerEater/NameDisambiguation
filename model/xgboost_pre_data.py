# encoding: utf-8

from cluster_model import *
import Levenshtein
import gensim
import numpy as np
from scipy import spatial
import json
import random
import file_config
from similarity import *

class XgbPreData(object):
    def __init__(self, vector_path):
        """
        :param vector_path:
        """

        # self.vec_dict = {}
        # self.vocab_set = set([])

        self.vec_dict = self.read_vec(vector_path)
        self.vocab_set = set(self.vec_dict.keys())


    def read_vec(self, vector_path):
        """
        读取词向量
        :param vector_path:
        :return:
        """
        vec_dict = {}

        with open(vector_path, "r") as vector_file:
            count = 0
            for item in vector_file:
                count += 1
                if count == 1:
                    continue

                item = item.strip().decode("utf-8")
                word = item.split(" ")[0]
                vector = np.array([float(ele) for ele in item.split(" ")[1:]])
                vec_dict[word] = vector
        return vec_dict

    def generate_train_test(self, origin_label_path, train_id_path, train_group_path, test_id_path, test_group_path):
        """
        构造训练集和测试集
        :param origin_label_path:
        :param train_id_path:
        :param train_group_path:
        :param test_id_path:
        :param test_group_path:
        :return:
        """
        origin_label = json.load(open(origin_label_path, "r"))

        doc_clusters = []
        for name, author_dict in origin_label.items():
            for author_id, doc_list in author_dict.items():
                doc_clusters.append(doc_list)

        # 按照簇长度排序
        doc_clusters = sorted(doc_clusters, key=lambda x: len(x), reverse=True)

        # 产生数据样例
        all_list = []
        for cluster_index, cluster in enumerate(doc_clusters):
            if len(cluster) < 5:
                break

            train_dict = {}
            # 产生正例，label为1， 同一个簇中产生
            for index in range(len(cluster)/4):
                pair_index = random.sample(range(len(cluster)), 2)
                pair_doc = [cluster[i] for i in pair_index]
                train_dict[" ".join(pair_doc)] = 1

            # 产生负例，label为2，和下一个簇产生
            next_cluster = doc_clusters[cluster_index + 1]
            for index in range(len(cluster)):
                first_ind = random.sample(range(len(cluster)), 1)[0]
                second_ind = random.sample(range(len(next_cluster)), 1)[0]
                pair_doc = [cluster[first_ind], next_cluster[second_ind]]
                train_dict[" ".join(pair_doc)] = 2

            all_list.append(train_dict)

        # 构造训练和测试集
        train_list = []
        test_list = []
        train_index_set = set(random.sample(range(len(all_list)), int(len(all_list)/5.0*4)))
        for i in range(len(all_list)):
            if i in train_index_set:
                train_list.append(all_list[i])
            else:
                test_list.append(all_list[i])

        with open(train_id_path, "w") as train_id_file:
            for item_dict in train_list:
                for pair_id, label in item_dict.items():
                    train_id_file.write(str(label).encode("utf-8") + "\t" +
                                        str(pair_id.split(" ")[0]).encode("utf-8")
                                        + "\t" + str(pair_id.split(" ")[1]) + "\n")

        with open(train_group_path, "w") as train_group_file:
            for item_dict in train_list:
                train_group_file.write(str(len(item_dict.keys())).encode("utf-8") + "\n")

        with open(test_id_path, "w") as test_id_file:
            for item_dict in test_list:
                for pair_id, label in item_dict.items():
                    test_id_file.write(str(label).encode("utf-8") + "\t" +
                                        str(pair_id.split(" ")[0]).encode("utf-8")
                                        + "\t" + str(pair_id.split(" ")[1]) + "\n")

        with open(test_group_path, "w") as test_group_file:
            for item_dict in test_list:
                test_group_file.write(str(len(item_dict.keys())).encode("utf-8") + "\n")

    def format_data(self, origin_data_path, data_id_path, group_path, format_data_path):
        """
        按照xgboost训练格式构造数据
        :param origin_data_path:
        :param data_id_path:
        :param group_path:
        :param format_data_path:
        :return:
        """
        origin_data = json.load(open(origin_data_path, "r"))

        group_list = []
        with open(group_path, "r") as group_file:
            for item in group_file:
                item = item.strip().decode("utf-8")
                group_list.append(int(item))

        qid = 0
        count = 0
        with open(data_id_path, "r") as data_file:
            with open(format_data_path, "w") as format_data_file:
                for item in data_file:
                    item = item.strip().decode("utf-8")
                    label, first_doc_id, second_doc_id = item.split("\t")

                    if origin_data.has_key(first_doc_id.split("-")[0]) and origin_data.has_key(second_doc_id.split("-")[0]):
                        first_doc = origin_data[first_doc_id.split("-")[0]]
                        first_author_index = int(first_doc_id.split("-")[1])
                        first_author_name = first_doc["authors"][first_author_index]["name"]
                        first_doc["q"] = first_doc_id

                        second_doc = origin_data[second_doc_id.split("-")[0]]
                        second_author_index = int(second_doc_id.split("-")[1])
                        second_author_name = second_doc["authors"][second_author_index]["name"]
                        second_doc["q"] = second_doc_id

                        features = similarity(Paper(1, first_author_name, first_doc), Paper(1, second_author_name, second_doc))
                        if len(features) < 12:
                            print "fea wrong"
                        fea_list = [str(fea) for fea in features]

                        format_data_file.write(str(label).encode("utf-8") + " qid:" + str(qid).encode("utf-8")
                                               + " " + " ".join(fea_list).encode("utf-8") + "\t" + "#" + "\t" +
                                               first_doc_id.encode("utf-8") + "\t" +
                                               second_doc_id.encode("utf-8") + "\n")

                        count += 1
                        if count == group_list[qid]:
                            qid += 1

                        if count % 1000 == 0:
                            print count


if __name__ == "__main__":
    xgb_pre = XgbPreData(file_config.vec_path)

    # xgb_pre.generate_train_test(file_config.external_v2_label, file_config.train_id_path, file_config.train_group_path,
    #                             file_config.test_id_path, file_config.test_group_path)

    xgb_pre.format_data(file_config.external_v2_data, file_config.train_id_path,
                        file_config.train_group_path, file_config.train_format_path)
