# encoding: utf-8

import json
import pre_config

import sys
reload(sys)
sys.setdefaultencoding("utf-8")


class FileUtil(object):
    def pre_word2vec(self, data_path_list, combine_path):
        """
        预处理数据词向量训练数据
        :param data_path_list: 参与训练词向量的数据列表
        :param combine_path: 结合后的数据
        :return:
        """
        with open(combine_path, "w") as combine_file:
            for data_path in data_path_list:
                with open(data_path, "r") as data_file:
                    load_dict = json.load(data_file)

                    for name, doc_list in load_dict.items():
                        for doc in doc_list:
                            if doc.has_key("title"):
                                combine_file.write(str(doc["title"]).encode("utf-8"))
                                combine_file.write("\n")

                            if doc.has_key("abstract"):
                                if doc["abstract"] != "":
                                    combine_file.write(str(doc["abstract"]).encode("utf-8"))
                                    combine_file.write("\n")

    def process_external_data(self, data_path, output_path):
        """
        读取外部文件中的信息
        :param data_path:
        :param output_path:
        :return:
        """
        with open(output_path, "w") as output_file:
            with open(data_path, "r") as data_file:
                for item in data_file:
                    item = item.strip().decode("utf-8")

                    paper = json.loads(item)

                    if paper.has_key("title"):
                        output_file.write(paper["title"].encode("utf-8"))
                        output_file.write("\n")

                    if paper.has_key("abstract"):
                        if paper["abstract"] != "":
                            output_file.write(paper["abstract"].encode("utf-8"))
                            output_file.write("\n")



    def combine_file(self, data_path_list, combine_path):
        """
        合并文件
        :param data_path_list:
        :param combine_path:
        :return:
        """
        with open(combine_path, "w") as combine_file:
            for data_path in data_path_list:
                with open(data_path, "r") as data_file:
                    for item in data_file:
                        item = item.strip().decode("utf-8")

                        combine_file.write(item.encode("utf-8") + "\n")



if __name__ == "__main__":
    file_util = FileUtil()

    data_path_list = [pre_config.train_validate_data_path, pre_config.test_sent_data_path]

    file_util.combine_file(data_path_list, pre_config.vec_sent_data_path)