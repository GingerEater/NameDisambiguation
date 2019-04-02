# coding: utf-8

import json
import Queue
import Levenshtein
import numpy as np
from scipy import spatial
import file_config
from xgboost_rank import *
from collections import Counter
from similarity import similarity, sent_distance, sent_ratio, sentence_sim, keyword_sim, vec_dict, vocab_set, is_mutual_sub
from utils import *
import multiprocessing
import os
from pprint import pprint

class Author(object):
    def __init__(self, json_author):
        # self.name = json_author['name'].lower()
        # self.org = json_author.get('org', '').lower()

        self.name = ' '.join([item.strip() for item in json_author['name'].lower().replace('-', ' ').split('.') if item != ''])
        self.org = org_process(json_author.get('org', ''))

    def original_info(self):
        return {'name': self.name, 'org': self.org}


class Paper(object):
    def __init__(self, index, name, json_paper):
        self.index = index
        self.cluster_index = index
        self.name = name
        self.id = json_paper.get('id', '')
        self.title = json_paper.get('title', '').lower()
        self.authors = [Author(json_author) for json_author in json_paper.get('authors', '')]
        self.venue = json_paper.get('venue', '').lower()
        self.year = json_paper.get('year', 0)
        self.keywords = [word.lower() for word in json_paper.get('keywords', [])]
        self.abstract = json_paper.get('abstract', '').lower()

        self.names = [author.name for author in self.authors]
        self.orgs = [author.org for author in self.authors]

        self.author_dic = dict(zip(self.names, self.orgs))
        self.org = self.author_dic.get(self.name, '')

        # 当前作者所处位置
        self.author_pos = 0
        for pos, author_name in enumerate(self.names):
            if author_name == name:
                self.author_pos = pos


    def original_info(self):
        # return self.__dict__  # self.authors 无法显示出来

        res = dict()
        res['id'] = self.id
        res['title'] = self.title
        res['authors'] = [author.original_info() for author in self.authors]
        res['venue'] = self.venue
        res['year'] = self.year
        res['keywords'] = self.keywords
        res['abstract'] = self.abstract
        return res


class Cluster(object):
    def __init__(self, index, name, json_cluster=None):
        self.index = index
        self.name = name
        # self.papers = [Paper(self.index, self.name, json_paper)]
        self.papers = []
        for json_paper in json_cluster:
            self.papers.append(Paper(self.index, self.name, json_paper))

        self.update_main()

        # self.main_paper = self.papers[0]
        # self.main_names = self.main_paper.names
        # self.main_org = self.main_paper.org

    def update_main(self):  # update after merge
        max_len = 0
        max_len_id = 0
        for i, paper in enumerate(self.papers):
            if max_len < len(paper.org.split()):
                max_len_id = i
                max_len = len(paper.org.split())

        self.main_paper = self.papers[max_len_id]
        self.main_names = self.main_paper.names
        self.main_org = self.main_paper.org
        self.main_venue = self.main_paper.venue
        self.main_title = self.main_paper.title
        self.main_keywords = self.main_paper.keywords
        self.index = self.main_paper.index

        for paper in self.papers:
            paper.cluster_index = self.index

    def output(self):
        return [paper.id for paper in self.papers]

    def original_info(self):
        return [paper.original_info() for paper in self.papers]


class Person(object):
    def __init__(self, name, json_person):
        self.name = name.lower()
        self.clusters = []

        for index, json_paper in enumerate(json_person):
            self.clusters.append(Cluster(index, self.name, [json_paper]))

        self.cluster_dict = {}
        for cluster in self.clusters:
            self.cluster_dict[cluster.index] = cluster

    def merge_cluster(self, cluster1_id, cluster2_id):
        cluster1 = self.cluster_dict[cluster1_id]
        cluster2 = self.cluster_dict[cluster2_id]

        cluster1.papers.extend(cluster2.papers)

        cluster1.update_main()
        self.cluster_dict[cluster1.index] = cluster1

        self.clusters.remove(cluster2)

    def remove_paper_from_cluster(self, cluster_item, paper):
        cluster_item.papers.remove(paper)

        cluster_item.update_main()
        self.cluster_dict[cluster_item.index] = cluster_item

        paper.cluster_index = paper.index
        new_cluster = Cluster(paper.index, self.name, [paper.original_info()])
        self.clusters.append(new_cluster)
        self.cluster_dict[new_cluster.index] = new_cluster


    def co_author_run(self):
        q = Queue.Queue(len(self.clusters))
        for cluster in self.clusters:
            q.put(cluster)

        while not q.empty():
            main_cluster = q.get()
            not_merge_clusters = []

            while not q.empty():
                cluster = q.get()

                # 含有缩写的name
                if len(main_cluster.main_paper.name.split(" ")[0]) == 1:
                    if (len(set(main_cluster.main_names) & set(cluster.main_names)) > 1 \
                        and (sent_distance(main_cluster.main_org, cluster.main_org) < 3 \
                             or is_mutual_sub(main_cluster.main_org, cluster.main_org))) \
                            or (len(set(main_cluster.main_names) & set(cluster.main_names)) > 2 \
                                and len(main_cluster.main_names) < 20 and len(cluster.main_names) < 20) \
                            or ((len(main_cluster.main_names) > 20 and len(cluster.main_names) > 20) \
                                and len(set(main_cluster.main_names) & set(cluster.main_names)) >
                                max(5, int(0.5 * min(len(main_cluster.main_names), len(cluster.main_names))))) \
                            or (len(set(main_cluster.main_names) & set(cluster.main_names)) > 1 \
                                and main_cluster.main_venue == cluster.main_venue and main_cluster.main_venue != "") \
                            or (len(set(main_cluster.main_names) & set(cluster.main_names)) > 1
                                and sentence_sim(main_cluster.main_title, cluster.main_title) > 0.7 \
                                and keyword_sim(main_cluster.main_keywords, cluster.main_keywords) > 0.7):

                        self.merge_cluster(main_cluster.index, cluster.index)
                    else:
                        not_merge_clusters.append(cluster)

                else:
                    if len(set(main_cluster.main_names) & set(cluster.main_names)) > 1 \
                            and (sent_distance(main_cluster.main_org, cluster.main_org) < 3 \
                                 or is_mutual_sub(main_cluster.main_org, cluster.main_org)):
                        self.merge_cluster(main_cluster.index, cluster.index)
                        continue

                    if len(set(main_cluster.main_names) & set(cluster.main_names)) > 2 \
                        and len(main_cluster.main_names) < 20 and len(cluster.main_names) < 20:
                        self.merge_cluster(main_cluster.index, cluster.index)
                        continue

                    if len(set(main_cluster.main_names) & set(cluster.main_names)) > 3:
                        self.merge_cluster(main_cluster.index, cluster.index)
                        continue

                    if len(set(main_cluster.main_names) & set(cluster.main_names)) > 1 \
                        and main_cluster.main_venue == cluster.main_venue and main_cluster.main_venue != "":
                        self.merge_cluster(main_cluster.index, cluster.index)
                        continue

                    if len(set(main_cluster.main_names) & set(cluster.main_names)) > 1 \
                        and sentence_sim(main_cluster.main_title, cluster.main_title) > 0.7 \
                        and keyword_sim(main_cluster.main_keywords, cluster.main_keywords) > 0.7:

                        self.merge_cluster(main_cluster.index, cluster.index)
                        continue

                    not_merge_clusters.append(cluster)


                    # if (len(set(main_cluster.main_names) & set(cluster.main_names)) > 1 \
                    #     and (sent_distance(main_cluster.main_org, cluster.main_org) < 4)) \
                    #         or (len(set(main_cluster.main_names) & set(cluster.main_names)) > 2 \
                    #             and len(main_cluster.main_names) < 20 and len(cluster.main_names) < 20) \
                    #         or (len(set(main_cluster.main_names) & set(cluster.main_names)) > 1 \
                    #             and main_cluster.main_venue == cluster.main_venue and main_cluster.main_venue != "") \
                    #         or (len(set(main_cluster.main_names) & set(cluster.main_names)) > 1
                    #             and sentence_sim(main_cluster.main_title, cluster.main_title) > 0.7
                    #             and ((keyword_sim(main_cluster.main_keywords, cluster.main_keywords) > 0.7
                    #             and len(main_cluster.main_keywords) != 0 and len(cluster.main_keywords) != 0)
                    #             or len(main_cluster.main_keywords) == 0 or len(cluster.main_keywords) == 0)):
                    #
                    #     self.merge_cluster(main_cluster.index, cluster.index)
                    # else:
                    #     not_merge_clusters.append(cluster)

            for cluster in not_merge_clusters:
                q.put(cluster)

    def co_author_second_run(self):
        q = Queue.Queue(len(self.clusters))
        for cluster in self.clusters:
            q.put(cluster)

        while not q.empty():
            main_cluster = q.get()
            not_merge_clusters = []

            while not q.empty():
                cluster = q.get()

                # 含有缩写的name
                if len(main_cluster.main_paper.name.split(" ")[0]) == 1:
                    pass
                else:
                    if self.is_author_same(main_cluster, cluster):
                        self.merge_cluster(main_cluster.index, cluster.index)
                    else:
                        not_merge_clusters.append(cluster)

            for cluster in not_merge_clusters:
                q.put(cluster)


    def org_run(self):
        no_information_num = 0
        q = Queue.Queue(len(self.clusters))
        for cluster in self.clusters:
            q.put(cluster)

        while not q.empty():
            main_cluster = q.get()
            while main_cluster.main_org == '' and not q.empty():
                main_cluster = q.get()
            not_merge_clusters = []

            while not q.empty():
                cluster = q.get()

                if cluster.main_org == '':
                    # print("Waring: no org information!!")
                    no_information_num += 1
                    continue

                # 含有缩写的name
                if len(main_cluster.main_paper.name.split(" ")[0]) == 1:
                    if sent_distance(main_cluster.main_org, cluster.main_org) < 3:
                        self.merge_cluster(main_cluster.index, cluster.index)
                    else:
                        not_merge_clusters.append(cluster)

                else:
                    if self.is_org_same(main_cluster.main_org, cluster.main_org) \
                            or self.is_org_author_same(main_cluster, cluster):
                        self.merge_cluster(main_cluster.index, cluster.index)
                    else:
                        not_merge_clusters.append(cluster)

            for cluster in not_merge_clusters:
                q.put(cluster)
        print("Number of no org information is:", no_information_num)


    def is_author_same(self, current_cluster, other_cluster):
        """
        判断两个簇是否有co_author大于2的paper
        :param current_cluster:
        :param other_cluster:
        :return:
        """
        is_merge = False

        for current_paper in current_cluster.papers:
            for other_paper in other_cluster.papers:
                if len(set(current_paper.names) & set(other_paper.names)) > 2 \
                        and len(current_paper.names) < 20 and len(other_paper.names) < 20:
                    is_merge = True
                    break

                if len(set(current_paper.names) & set(other_paper.names)) == 3 \
                    and (sent_distance(current_paper.org, other_paper.org) < 3 \
                         or is_mutual_sub(current_paper.org, other_paper.org) \
                         or current_paper.venue == other_paper.venue):
                    is_merge = True
                    break

                if len(set(current_paper.names) & set(other_paper.names)) > 3:
                    is_merge = True
                    break

            if is_merge:
                break

        return is_merge

    def is_org_author_same(self, current_cluster, other_cluster):
        """
        判断当前两个cluster中是否有org相同且co_author也相同的paper
        :param current_cluster:
        :param other_cluster:
        :return:
        """
        is_merge = False

        for current_paper in current_cluster.papers:
            for other_paper in other_cluster.papers:

                if sent_distance(current_paper.org, other_paper.org) < 3 \
                        and len(set(current_paper.names) & set(other_paper.names)) > 1 \
                        and max(len(current_cluster.papers), len(other_cluster.papers)) < 100 \
                        and len(current_cluster.papers) * len(other_cluster.papers) < 200:

                    is_merge = True
                    break

                if sent_distance(current_paper.org, other_paper.org) < 3 \
                        and len(set(current_paper.names) & set(other_paper.names)) > 1 \
                        and len(current_paper.org.split()) > 4:

                    # print("is_org_author_same2")
                    # print(current_paper.org, other_paper.org)
                    # print(len(set(current_paper.names) & set(other_paper.names)))
                    # print(current_cluster.main_title)
                    # print(other_cluster.main_title)
                    # print(len(current_cluster.papers), len(other_cluster.papers))
                    # print("\n" * 2)

                    is_merge = True
                    break

            if is_merge:
                break

        return is_merge

    def is_org_same(self, current_org, other_org):
        """
        判断org是否相等
        :param current_cluster:
        :param other_cluster:
        :return:
        """
        is_org = False

        if sent_distance(current_org, other_org) < 3 \
            and ((len(current_org.split()) > 3 \
                 and len(other_org.split()) > 3)
                 or is_org_contain_keyword(current_org)):

            is_org = True

        return is_org

    def combine_cluster(self):
        """
        合并现有的簇
        :param xgboost_model:
        :return:
        """
        q = Queue.Queue(len(self.clusters))

        # 对当前簇按照从大到小顺序进行排序
        cluster_dict = {}
        for index, cluster in enumerate(self.clusters):
            cluster_dict[index] = [len(cluster.papers), cluster]
        sort_cluster_list = sorted(cluster_dict.items(), key=lambda x: x[1][0], reverse=True)
        sort_cluster_list = [cluster_pair[1][1] for cluster_pair in sort_cluster_list]

        for cluster_item in sort_cluster_list:
            q.put(cluster_item)

        while not q.empty():
            current_cluster = q.get()
            # 单篇文章的簇不合并
            while len(current_cluster.papers) == 1 and not q.empty():
                current_cluster = q.get()
            not_merge_clusters = []

            while not q.empty():
                other_cluster = q.get()

                # 不考虑与单篇文章的簇合并
                if len(other_cluster.papers) == 1:
                    continue

                short_org_same_num, long_org_same_num, same_org_num, same_org_ratio, \
                co_author_num, same_venue_num = self.cluster_sim(current_cluster, other_cluster)

                # 含有缩写的name
                if len(current_cluster.main_paper.name.split(" ")[0]) == 1:
                    if same_org_num > 3 or same_org_ratio > 0.4:

                        self.merge_cluster(current_cluster.index, other_cluster.index)
                    else:
                        not_merge_clusters.append(other_cluster)
                else:
                    if long_org_same_num > 3:

                        self.merge_cluster(current_cluster.index, other_cluster.index)
                        continue

                    if short_org_same_num > 3 and co_author_num > 2 \
                            and (len(current_cluster.papers) < 20 or len(other_cluster.papers) < 20) \
                            and (float(short_org_same_num)/len(current_cluster.papers) > 0.15 \
                                 or float(co_author_num)/len(current_cluster.papers) > 0.15 \
                                 or short_org_same_num > len(current_cluster.papers)):

                        self.merge_cluster(current_cluster.index, other_cluster.index)
                        continue

                    if short_org_same_num > 3 and co_author_num > 10 \
                            and (len(current_cluster.papers) > 20 and len(other_cluster.papers) > 20):

                        self.merge_cluster(current_cluster.index, other_cluster.index)
                        continue

                    if same_org_ratio > 0.4 and len(other_cluster.papers) < 9 \
                            and len(current_cluster.papers) * len(other_cluster.papers) < 300:
                        # print("same_org_ratio")
                        # print("short_org_same_num:{0}".format(short_org_same_num))
                        # print("long_org_same_num:{0}".format(long_org_same_num))
                        # print("co_author_num:{0}".format(co_author_num))
                        # print(len(current_cluster.papers))
                        # print(len(other_cluster.papers))
                        # print(current_cluster.main_org)
                        # print(other_cluster.main_org)
                        # print("\n" * 3)

                        self.merge_cluster(current_cluster.index, other_cluster.index)
                        continue


                    else:
                        not_merge_clusters.append(other_cluster)

                    # if same_org_num > 3 or same_org_ratio > 0.4 \
                    #         or (co_author_num > 3 and same_venue_num > 1 \
                    #             and max(len(current_cluster.papers), len(other_cluster.papers)) < 150):
                    #         # or (co_author_num > 2 and same_org_num > 2 \
                    #         #     and max(len(current_cluster.papers), len(other_cluster.papers)) < 100) \
                    #         # or (co_author_num > 2 and same_venue_num > 1 and same_org_num > 1 \
                    #         #     and max(len(current_cluster.papers), len(other_cluster.papers)) < 100) \
                    #         # or (same_venue_num > 1 and same_org_num > 2 \
                    #         #     and len(current_cluster.papers) < 40 and len(other_cluster.papers) < 40) \
                    #         # or (len(current_cluster.papers) < 7 and len(other_cluster.papers) < 7 \
                    #         #     and ((co_author_num > 1 and same_venue_num > 0) \
                    #         #          or (co_author_num > 1 and same_org_num > 0) \
                    #         #          or (same_org_num > 1 and same_venue_num > 0))):
                    #     # print "cluster index:{0}, {1}".format(current_cluster.index, other_cluster.index)
                    #     self.merge_cluster(current_cluster.index, other_cluster.index)
                    # else:
                    #     not_merge_clusters.append(other_cluster)

            for cluster in not_merge_clusters:
                q.put(cluster)

    def cluster_sim(self, current_cluster, other_cluster):
        """
        计算cluster之间org重合度
        :param current_cluster:
        :param other_cluster:
        :return:
        """
        current_org_list = []
        current_venue_set = set()
        for paper in current_cluster.papers:
            if paper.org != "":
                current_org_list.append(paper.org)
            if paper.venue != "":
                current_venue_set.add(paper.venue)

        other_org_list = []
        other_venue_set = set()
        for paper in other_cluster.papers:
            if paper.org != "":
                other_org_list.append(paper.org)
            if paper.venue != "":
                other_venue_set.add(paper.venue)

        same_org_ratio = 0.0
        short_org_same_num = 0
        long_org_same_num = 0
        same_org_num = 0
        co_author_num = 0
        same_venue_num = 0

        # 对文章数多的簇遍历
        if len(current_cluster.papers) >= len(other_cluster.papers):
            other_org_set = set(other_org_list)
            for current_org in current_org_list:
                if (len(current_org.split()) < 4 and not is_org_contain_keyword(current_org)) and current_org in other_org_set:
                    short_org_same_num += 1
                    continue

                if (len(current_org.split()) > 3 or is_org_contain_keyword(current_org)) and current_org in other_org_set:
                    if is_org_special(current_org) and current_cluster.name == "meng wang":
                        short_org_same_num += 1
                    else:
                        long_org_same_num += 1

            for current_paper in current_cluster.papers:
                if current_paper.venue in other_venue_set:
                    same_venue_num += 1

                for other_paper in other_cluster.papers:
                    if len(set(current_paper.names) & set(other_paper.names)) > 1 \
                            and abs(len(current_paper.names) - len(other_paper.names)) < 8:

                        co_author_num += 1

        else:
            current_org_set = set(current_org_list)
            for other_org in other_org_list:
                if (len(other_org.split()) < 4 and not is_org_contain_keyword(other_org)) and other_org in current_org_set:
                    short_org_same_num += 1
                    continue


                if (len(other_org.split()) > 3 or is_org_contain_keyword(other_org)) and other_org in current_org_set:
                    if is_org_special(other_org) and current_cluster.name == "meng wang":
                        short_org_same_num += 1
                    else:
                        long_org_same_num += 1

            for other_paper in other_cluster.papers:
                if other_paper.venue in current_venue_set:
                    same_venue_num += 1

                for current_paper in current_cluster.papers:
                    if len(set(current_paper.names) & set(other_paper.names)) > 1 \
                            and abs(len(current_paper.names) - len(other_paper.names)) < 8:
                        co_author_num += 1

        same_org_num = short_org_same_num + long_org_same_num
        if len(current_org_list) + len(other_org_list) != 0:
            same_org_ratio = float(same_org_num) / \
                             (len(current_org_list) + len(other_org_list))

        return [short_org_same_num, long_org_same_num, same_org_num, same_org_ratio, co_author_num, same_venue_num]

    def combine_small(self):
        """
        合并小簇
        :return:
        """
        single_paper_clusters = [cluster for cluster in self.clusters if len(cluster.papers) == 1]
        not_single_paper_clusters = [cluster for cluster in self.clusters if len(cluster.papers) > 1]
        not_single_paper_clusters = sorted(not_single_paper_clusters, key=lambda x: len(x.papers))
        small_paper_clusters = [cluster for cluster in self.clusters if len(cluster.papers) < 5]

        for single_cluster in single_paper_clusters:
            is_merged = False
            main_paper = single_cluster.papers[0]
            for not_single_cluster in not_single_paper_clusters:
                # sing_paper_cluster.org 和 大 cluster.main_org
                if sent_distance(main_paper.org, not_single_cluster.main_org) < 3 \
                        and len(main_paper.org.split()) > 3:
                    self.merge_cluster(not_single_cluster.index, single_cluster.index)
                    break

                if len(single_cluster.name.split()[0]) == 1:
                    continue

                for paper in not_single_cluster.papers:
                    # 根据 co_author 合并单个的簇与大
                    if len(set(main_paper.names) & set(paper.names)) > 1 \
                            and ((sentence_sim(main_paper.title, paper.title) > 0.7 \
                            and keyword_sim(main_paper.keywords, paper.keywords) > 0.7) \
                                 or (main_paper.venue == paper.venue and main_paper.venue != "")):
                        self.merge_cluster(not_single_cluster.index, single_cluster.index)
                        is_merged = True
                        break

                if is_merged:
                    break


    def combine_other(self, xgboost_model):
        """
        将other中的cluster加入到大簇中
        :param xgboost_model:
        :return:
        """
        tmp_clusters = [cluster for cluster in self.clusters]

        # 读取所有main_paper
        main_papers = [cluster.main_paper for cluster in tmp_clusters]

        count = 0
        for cluster in tmp_clusters:
            if len(self.cluster_dict[cluster.index].papers) == 1:
                # count += 1
                # if count % 50 == 0:
                #     print count

                current_paper = cluster.papers[0]
                other_paper_list = []
                feas_list = []
                for paper in main_papers:
                    if paper.id != current_paper.id:
                        other_paper_list.append(paper)
                        # 计算两篇paper相似度
                        feas = similarity(current_paper, paper)
                        feas_list.append(feas)

                dtest = xgb.DMatrix(feas_list)
                dtest.set_group([len(feas_list)])
                preds = xgboost_model.predict(dtest).tolist()

                # 统计前3个paper对应的cluster
                pred_dict = {}
                for i, val in enumerate(preds):
                    pred_dict[i] = val

                sort_pred_list = sorted(pred_dict.items(), key=lambda x: x[1])
                # pred_index_list = [ele[0] for ele in sort_pred_list[:3]]
                # pre_fea_list = [feas_list[index] for index in pred_index_list]
                # pred_cluster_indexs = [other_paper_list[index].cluster_index for index in pred_index_list]

                # 合并前过滤
                if (current_paper.org != "" and feas_list[sort_pred_list[0][0]][0] < 0.5 and sort_pred_list[0][1] > -2.8) \
                        or (current_paper.org == "" and sort_pred_list[0][1] > -2.3):
                    # print "nocombine:{0}, fea:{1}".format(sort_pred_list[0][1], feas_list[sort_pred_list[0][0]])
                    continue

                # print "combine:{0}, fea:{1}".format(sort_pred_list[0][1], feas_list[sort_pred_list[0][0]])

                # print "cluster index:{0}, {1}".format(other_paper_list[sort_pred_list[0][0]].cluster_index, cluster.index)
                self.merge_cluster(other_paper_list[sort_pred_list[0][0]].cluster_index, cluster.index)

                # 选择当前paper应该加入的簇
                # if len(set(pred_cluster_indexs)) < len(pred_cluster_indexs):
                #     self.merge_cluster(cluster_dict[Counter(pred_cluster_indexs).most_common(1)[0][0]], cluster)
                # else:
                #     self.merge_cluster(cluster_dict[pred_cluster_indexs[0]], cluster)


    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        idx = 0
        for cluster in self.clusters:
            if len(cluster.papers) > 1:
                res = cluster.original_info()
                # json.dump(res, open('tmp_cluster/' + str(len(cluster.papers)) + '_' + str(idx) + '.json', 'w'), indent=4)
                json.dump(res, open(path + str(len(cluster.papers)) + '_' + str(idx) + '.json', 'w'), indent=4)
                idx += 1
        res = [cluster.original_info() for cluster in self.clusters if len(cluster.papers) == 1]
        json.dump(res, open(path + '1.json', 'w'), indent=4)


def worker(i, name):
    print(i)
    print(name)
    person = Person(' '.join(name.split('_')), valid_data[name])
    print(len(person.clusters))
    person.co_author_run()
    person.co_author_second_run()
    person.org_run()
    person.combine_cluster()
    person.combine_cluster()
    person.combine_small()
    person.combine_other(rank_model)
    print(len(person.clusters))
    print(sorted(Counter([len(cluster.papers) for cluster in person.clusters]).items(), key=lambda x: x[0]))

    res = {}
    res[name] = [cluster.output() for cluster in person.clusters]
    json.dump(res, open('result_dir/result' + str(i) + '.json', 'w'), indent=4)

if __name__ == '__main__':
    # data = json.load(open('/Users/coder352/datasets/Entity/Name_Disambiguation/Scholar2018/pubs_train.json'))
    # test_data = json.load(open(file_config.test_data_path))

    # xgboost_rank = XgboostRank(file_config.xgboost_model_path)
    # rank_model_model = xgboost_rank.load_rank_model()

    # person = Person('shuang li', test_data['shuang_li'])
    # print(len(person.clusters))
    #
    # print(person.name)
    # print(person.clusters[0].papers[0].title)
    # print(person.clusters[0].main_paper.id)
    #
    # person.co_author_run()
    # print(len(person.clusters))
    # print(sorted(Counter([len(cluster.papers) for cluster in person.clusters]).items(), key=lambda x: x[0]))
    # person.save('tmp_cluster/' + person.name + '/co_author_run/')
    #
    # person.co_author_second_run()
    # print(len(person.clusters))
    # print(sorted(Counter([len(cluster.papers) for cluster in person.clusters]).items(), key=lambda x: x[0]))
    # person.save('tmp_cluster/' + person.name + '/co_author_second_run/')
    #
    # person.org_run()
    # print(len(person.clusters))
    # print(sorted(Counter([len(cluster.papers) for cluster in person.clusters]).items(), key=lambda x: x[0]))
    # person.save('tmp_cluster/' + person.name + '/org_run/')
    #
    # person.combine_cluster()
    # print(len(person.clusters))
    # print(sorted(Counter([len(cluster.papers) for cluster in person.clusters]).items(), key=lambda x: x[0]))
    # person.save('tmp_cluster/' + person.name + '/combine_cluster1/')
    #
    # person.combine_cluster()
    # print(len(person.clusters))
    # print(sorted(Counter([len(cluster.papers) for cluster in person.clusters]).items(), key=lambda x: x[0]))
    # person.save('tmp_cluster/' + person.name + '/combine_cluster2/')
    #
    # person.combine_small()
    # print(len(person.clusters))
    # print(sorted(Counter([len(cluster.papers) for cluster in person.clusters]).items(), key=lambda x: x[0]))
    # person.save('tmp_cluster/' + person.name + '/combine_small/')

    # person.combine_other(rank_model_model)
    # print(len(person.clusters))
    # print(sorted(Counter([len(cluster.papers) for cluster in person.clusters]).items(), key=lambda x: x[0]))


    ################################################################
    # write one cluster into one file
    # import os
    # if not os.path.exists("tmp_cluster/"):
    #     os.makedirs("tmp_cluster/")
    # idx = 0
    # for cluster in person.clusters:
    #     res = cluster.original_info()
    #     # json.dump(res, open('tmp_cluster/' + str(len(cluster.papers)) + '_' + str(cluster.index) + "_" + str(idx) + '.json', 'w'), indent=4)
    #
    #     if len(cluster.papers) > 1:
    #         json.dump(res, open('tmp_cluster/' + str(len(cluster.papers)) + '_' + str(idx) + '.json', 'w'), indent=4)
    #         idx += 1
    #
    # res = [cluster.original_info() for cluster in person.clusters if len(cluster.papers) == 1]
    # json.dump(res, open('tmp_cluster/1.json', 'w'), indent=4)

    ##################################################################
    # For test data
    xgboost_rank = XgboostRank(file_config.xgboost_model_path)
    rank_model = xgboost_rank.load_rank_model()
    test_data = json.load(open(file_config.test_data_path))
    res = {}

    for i, name in enumerate(test_data.keys()):
        person = Person(' '.join(name.split('_')), test_data[name])

        print(i)
        print(person.name)
        print(len(person.clusters))
        print(person.clusters[0].main_paper.id)

        person.co_author_run()
        print(len(person.clusters))
        print(sorted(Counter([len(cluster.papers) for cluster in person.clusters]).items(), key=lambda x: x[0]))

        person.co_author_second_run()
        print(len(person.clusters))
        print(sorted(Counter([len(cluster.papers) for cluster in person.clusters]).items(), key=lambda x: x[0]))

        person.org_run()
        print(len(person.clusters))
        print(sorted(Counter([len(cluster.papers) for cluster in person.clusters]).items(), key=lambda x: x[0]))

        person.combine_cluster()
        print(len(person.clusters))
        print(sorted(Counter([len(cluster.papers) for cluster in person.clusters]).items(), key=lambda x: x[0]))

        person.combine_cluster()
        print(len(person.clusters))
        print(sorted(Counter([len(cluster.papers) for cluster in person.clusters]).items(), key=lambda x: x[0]))

        person.combine_small()
        print(len(person.clusters))
        print(sorted(Counter([len(cluster.papers) for cluster in person.clusters]).items(), key=lambda x: x[0]))

        person.combine_other(rank_model)
        print(len(person.clusters))
        print(sorted(Counter([len(cluster.papers) for cluster in person.clusters]).items(), key=lambda x: x[0]))

        res[name] = [cluster.output() for cluster in person.clusters]

    json.dump(res, open('result.json', 'w'), indent=4)

    ##################################################################
    # for Valid data
    # if not os.path.exists("result_dir/"):
    #     os.makedirs("result_dir/")
    # valid_data = json.load(open(file_config.validate_data_path))
    # xgboost_rank = XgboostRank(file_config.xgboost_model_path)
    # rank_model = xgboost_rank.load_rank_model()
    #
    # for i, name in enumerate(valid_data.keys()):
    #     p = multiprocessing.Process(target=worker, args=(i, name))
    #     p.start()

    # res = {}
    # for i in range(50):
    #     name_json = json.load(open('result_dir/result' + str(i) + '.json'))
    #     res[list(name_json)[0]] = name_json[list(name_json)[0]]
    #
    # json.dump(res, open('result.json', 'w'), indent=4)

