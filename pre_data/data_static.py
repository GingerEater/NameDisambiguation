# encoding: utf-8


import pre_config
import json
from collections import Counter
from pprint import pprint

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

class DataStatic(object):

    def people_static(self, data_path):
        """
        统计人数
        :return:
        """
        with open(data_path, "r") as data_file:
            load_dict = json.load(data_file)

            print "name mention num:{0}".format(len(load_dict.keys()))

            doc_num = 0
            author_num = 0
            author_org_num = 0
            title_num = 0
            abstract_num = 0
            keywords_num = 0
            venue_num = 0
            year_num = 0

            for name, doc_list in load_dict.items():
                print name + "\t" + str(len(doc_list))

                doc_num += len(doc_list)

                for doc in doc_list:
                    author_list = doc["authors"]
                    author_num += len(author_list)

                    for author in author_list:
                        if author.has_key("org"):
                            if author["org"] != "":
                                author_org_num += 1

                    if doc.has_key("title"):
                        title_num += 1

                    if doc.has_key("abstract"):
                        if doc["abstract"] != "":
                            abstract_num += 1

                    if doc.has_key("keywords"):
                        if len(doc["keywords"]) > 0:
                            keywords_num += 1

                    if doc.has_key("venue"):
                        if doc["venue"] != "":
                            venue_num += 1

                    if doc.has_key("year"):
                        if doc["year"] != "":
                            year_num += 1



            # print "doc num:{0}\nauthor num:{1}\nauthor_org num:{2}\n" \
            #       "title num:{3}\nabstract num:{4}\nkeywords num:{5}\n" \
            #       "venue num:{6}\nyear num:{7}".format(doc_num, author_num,
            #                                             author_org_num, title_num,
            #                                             abstract_num, keywords_num, venue_num, year_num)

    def org_static(self, data_path):
        """
        :param data_path:
        :return:
        """
        load_dict = json.load(open(data_path, "r"))

        org_word_list = []
        for name, doc_list in load_dict.items():
            for doc in doc_list:
                author_list = doc["authors"]

                for author in author_list:
                    if author.has_key("org"):
                        if author["org"] != "":
                            org_word_list.extend(author["org"].split(" "))

        with open("org_word_static", "w") as org_word_file:
            org_words = sorted(Counter([word.lower() for word in org_word_list]).items(), key=lambda x: x[1], reverse=True)
            for word_turp in org_words:
                org_word_file.write(str(word_turp[0]).encode("utf-8") + "\t" + str(word_turp[1]).encode("utf-8") + "\n")

    def venue_static(self, data_path):
        """
        :param data_path:
        :return:
        """
        load_dict = json.load(open(data_path, "r"))

        for item in load_dict.keys():
            print item


        # venue_word_list = []
        # for name, doc_list in load_dict.items():
        #     for doc in doc_list:
        #         if doc.has_key("venue"):
        #             if doc["venue"] != "":
        #                 venue_word_list.extend(doc["venue"].split(" "))
        #
        # with open("venue_word_static", "w") as venue_word_file:
        #     venue_words = sorted(Counter([word.lower() for word in venue_word_list]).items(), key=lambda x: x[1], reverse=True)
        #     for word_turp in venue_words:
        #         venue_word_file.write(str(word_turp[0]).encode("utf-8") + "\t" + str(word_turp[1]).encode("utf-8") + "\n")

    def groud_truth_static(self, data_path):
        """
        统计分析groud truth
        :return:
        """
        load_dict = json.load(open(data_path, "r"))

        for name, clsuter_list in load_dict.items():
            print(name)
            print(sorted([len(cluster) for cluster in clsuter_list], reverse=True))
            print("\n"*2)



if __name__ == "__main__":
    data_static = DataStatic()

    data_static.groud_truth_static(pre_config.validate_groud_truth_path)
