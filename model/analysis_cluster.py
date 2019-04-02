#!/usr/bin/python3
# coding: utf-8
import glob
import json
from collections import Counter
from pprint import pprint
from cluster_model import Author, Paper, Cluster, Person
import os

def analysis_cluster(cluster1, cluster2):
    print(cluster1.main_names)
    print(cluster2.main_names)
    print(cluster1.main_org)
    print(cluster2.main_org)

    same_keyword_ratio, same_title_word_ratio, same_venue_ratio, same_org_ratio, same_keyword_num, same_venue_num, \
    same_org_num, cluster_keyword_similarity, main_paper_feas = Person.pair_cluster_static(cluster1, cluster2)

    print('same_keyword_ratio:', same_keyword_ratio)
    print('same_title_word_ratio:', same_title_word_ratio)
    print('same_venue_ratio:', same_venue_ratio)
    print('same_org_ratio:', same_org_ratio)
    print('same_keyword_num:', same_keyword_num)
    print('same_venue_num:', same_venue_num)
    print('same_org_num:', same_org_num)
    print('cluster_keyword_similarity:', cluster_keyword_similarity)
    print('main_paper_feas:', main_paper_feas)

def analysis_one_cluster(cluster):
    """
    分析单点簇
    :param cluster1:
    :return:
    """
    print("main_org" + "\t" + cluster.main_org + "\n")

    org_list = []
    for paper in cluster.papers:
        org_list.append(paper.org)

    # print("org set:")
    # org_set = set(org_list)
    # print("\n".join(list(org_set)))

    print("org list count:")
    for pair in sorted(Counter(org_list).items(), key=lambda x: x[1], reverse=True):
        print(pair)
    print("\n")


if __name__ == '__main__':
    dir_path = './tmp_cluster/shuang li/co_author_run/'

    # file_name = dir_path + "31_31.json"
    # cluster1 = Cluster(1, 'meng wang', json.load(open(file_name)))
    # analysis_one_cluster(cluster1)


    file_list = os.listdir(dir_path)  # 列出文件夹下所有的目录与文件

    for i in range(len(file_list)):
        if str(file_list[i]) == '1.json':
            continue

        file_path = os.path.join(dir_path, file_list[i])
        print(file_list[i])

        if os.path.isfile(file_path):
            cluster1 = Cluster(1, 'shuang li', json.load(open(file_path)))
            analysis_one_cluster(cluster1)
            # print("\n"*4)
