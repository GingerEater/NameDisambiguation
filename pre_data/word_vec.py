# encoding:utf-8

import gensim
import pre_config



class MySentences(object):
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        sent_seg_file = open(self.path, "r")

        for item in sent_seg_file:
            item = item.decode("utf-8").strip()
            sent = item.split(" ")
            yield sent

class Word2Vec(object):
    """
    训练词向量类
    """

    def train(self, data_path, model_path):
        """
        训练词向量
        :return:
        """
        model = gensim.models.Word2Vec(MySentences(data_path), min_count=2, size=50)
        model.save(model_path)


    def online_train(self, model_path, data_path):
        """
        增量训练，不会增加新的词向量，只会对原有的词向量进行修改
        :return:
        """
        model = gensim.models.Word2Vec.load(model_path)
        model.train(MySentences(data_path), total_examples=model.corpus_count, epochs=model.iter)

    def get_vector(self, model_path, vector_path, vocab_path):
        """
        获取训练的词向量及字典并将其存储到文件中
        :return:
        """
        model = gensim.models.Word2Vec.load(model_path)

        model.wv.save_word2vec_format(vector_path, fvocab=vocab_path)


    def test_vector(self, model_path, test_pair_list):
        """
        测试词向量的质量
        :param model_path:
        :param test_pair_list:
        :return:
        """
        model = gensim.models.Word2Vec.load(model_path)

        for test_pair in test_pair_list:
            pair_sim = model.similarity(test_pair[0], test_pair[1])
            print "{0}, {1}: {2}".format(test_pair[0], test_pair[1], pair_sim)




if __name__ == "__main__":
    word2vec = Word2Vec()


    test_pair_list = []
    test_pair_list.append(("chromosome", "genes"))
    test_pair_list.append(("chromosome", "allele"))
    test_pair_list.append(("laparoscope", "prostate"))
    test_pair_list.append(("bladder", "prostate"))
    test_pair_list.append(("surface", "photometric"))
    test_pair_list.append(("bladder", "photometric"))
    test_pair_list.append(("feature", "extraction"))
    test_pair_list.append(("router", "protocol"))
    test_pair_list.append(("bladder", "router"))

    word2vec.test_vector(pre_config.word2vec_model_path, test_pair_list)

