#!/usr/bin/python3
# coding: utf-8
import json
import Levenshtein
import gensim
import numpy as np
from scipy import spatial
import file_config
import re

def read_vec(vector_path):
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

def load_filter_word(word_path):
    """
    过滤org中常见词
    :param word_path:
    :return:
    """
    filter_word_set = set()
    with open(word_path, "r") as org_filter_word_file:
        for item in org_filter_word_file:
            item = item.strip().decode("utf-8")
            word = item.split("\t")[0]
            filter_word_set.add(word)

    return list(filter_word_set)

vec_dict = read_vec(file_config.vec_path)
vocab_set = set(vec_dict.keys())



def sentence_sim(sen_first, sen_second):
    global vec_dict
    global vocab_set

    sen_sim = 0.0
    sen_first_vec = [vec_dict[word] for word in sen_first.split(' ') if
                      word in vocab_set]
    sen_second_vec = [vec_dict[word] for word in sen_second.split(' ') if
                 word in vocab_set]

    if len(sen_first_vec) > 0 and len(sen_second_vec) > 0:
        sen_first_vec = np.mean(sen_first_vec, axis=0)
        sen_second_vec = np.mean(sen_second_vec, axis=0)
        sen_sim = 1 - spatial.distance.cosine(sen_first_vec, sen_second_vec)

    return sen_sim

def keyword_sim(main_keywords, other_keywords):
    global vec_dict
    global vocab_set

    keyword_semantic_similarity = 0.0
    main_keyword_vec = [vec_dict[word] for item in main_keywords for word in item.split(' ')
                        if word in vocab_set]
    keyword_vec = [vec_dict[word] for item in other_keywords for word in item.split(' ') if
                   word in vocab_set]
    if len(main_keyword_vec) > 0 and len(keyword_vec) > 0:
        main_keyword_vec = np.mean(main_keyword_vec, axis=0)
        keyword_vec = np.mean(keyword_vec, axis=0)
        keyword_semantic_similarity = 1 - spatial.distance.cosine(main_keyword_vec, keyword_vec)

    return keyword_semantic_similarity

def sent_ratio(sent_first, sent_second):
    sent_ratio = 0.0
    if sent_first != "" and sent_second != "":
        sent_ratio = Levenshtein.ratio(sent_first, sent_second)

    return sent_ratio

def sent_distance(sent_first, sent_second):
    sent_dist = 100
    if sent_first != "" and sent_second != "":
        sent_dist = Levenshtein.distance(sent_first.strip(), sent_second.strip())

    return sent_dist

def is_mutual_sub(sent_first, sent_second):
    is_sub = False
    if sent_first != "" and sent_second != "":
        if sent_first in sent_second or sent_second in sent_first:
            is_sub = True

    return is_sub

def similarity(main_paper, paper):
    global vec_dict
    global vocab_set

    # org ratio
    org_ratio = sent_ratio(main_paper.org, paper.org)

    # 论文发表年份之差
    year_dist = 0
    if main_paper.year != 0 and paper.year != 0:
        year_dist = abs(main_paper.year - paper.year)

    # 作者数量之差
    author_num_dist = abs(len(main_paper.names) - len(paper.names))

    # 作者位置之差
    author_pos_dist = abs(main_paper.author_pos - paper.author_pos)

    # 期刊ratio
    venue_ratio = sent_ratio(main_paper.venue, paper.venue)

    # 期刊语义相似度
    venue_semantic_similarity = sentence_sim(main_paper.venue, paper.venue)

    # title ratio
    title_ratio = sent_ratio(main_paper.title, paper.title)

    title_same_word_num = len(set(main_paper.title.split()) & set(paper.title.split()))

    # 标题词数之差
    title_num_dist = abs(len(main_paper.title.split()) - len(paper.title.split()))

    # 标题语义相似度
    title_semantic_similarity = sentence_sim(main_paper.title, paper.title)

    # 关键词重合数
    keyword_same_word_num = len(
        set([item for item in main_paper.keywords]) & set([item for item in paper.keywords]))

    # 关键词语义相似度
    keyword_semantic_similarity = keyword_sim(main_paper.keywords, paper.keywords)

    return [org_ratio, venue_ratio, title_ratio, year_dist, author_num_dist, author_pos_dist,
            title_num_dist, title_same_word_num, keyword_same_word_num, title_semantic_similarity,
            venue_semantic_similarity, keyword_semantic_similarity]



if __name__ == '__main__':

    print sentence_sim("",
                       "")