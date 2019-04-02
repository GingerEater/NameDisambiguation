# encoding: utf-8

import json


class Validation:
    precision = recall = f1 = 0.
    name_count = 0

    def __init__(self, precision=0., recall=0., f1=0., name_count=0.):
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.name_count = name_count

    # 统计同一名称下的正确率, 召回率, f1 值
    # truths 是一个字典, key 是文章 id, value 是作者 id
    # preds 是一个字典,  key 是文章 id, value 是簇 id
    def pairwise_count(self, preds, truths):
        # TP 表示应该聚成一类的成对作者被正确聚成一类的总数
        # FP 表示不应该聚成一类的成对作者被错误聚成一类的总数
        # FN 表示应该聚成一类的成对作者被错误分开的总数
        tp_name = fp_name = fn_name = 0
        # 验证数据
        pred_set = set(preds.keys())
        truth_set = set(truths.keys())
        if (not (len(pred_set - truth_set) == 0 and len(truth_set - pred_set) == 0)):
            print("文章 id 不对应! ")
            return 0, 0, 0

        values = list(truths.keys())
        n_samples = len(values)
        for i in range(n_samples - 1):
            pred_i = preds[values[i]]
            for j in range(i + 1, n_samples):
                pred_j = preds[values[j]]
                if pred_i == pred_j:
                    if truths[values[i]] == truths[values[j]]:
                        tp_name += 1
                    else:
                        fp_name += 1
                elif truths[values[i]] == truths[values[j]]:
                    fn_name += 1
        return self.cal_p_r_f1(tp_name, fp_name, fn_name)

    # 计算总体正确率, 召回率和 f1 值
    def cal_p_r_f1(self, tp, fp, fn):
        tp_plus_fp = tp + fp
        tp_plus_fn = tp + fn
        if tp_plus_fp == 0:
            precision = 0.
        else:
            precision = tp / tp_plus_fp
        if tp_plus_fn == 0:
            recall = 0.
        else:
            recall = tp / tp_plus_fn
        if not precision or not recall:
            f1 = 0.
        else:
            f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    def reset(self):
        self.precision = 0.
        self.recall = 0.
        self.f1 = 0.
        self.name_count = 0

    # 验证 na 数据集
    def validate_na(self, truth_na_file, preds_na_file):
        with open(truth_na_file, "r", encoding='utf-8') as f_grand_truth:
            truth = json.loads(str(f_grand_truth.read()))
            print(truth_na_file + "   加载文件完成...")
        with open(preds_na_file, "r", encoding='utf-8') as f_preds:
            preds = json.loads(str(f_preds.read()))
            print(preds_na_file + "   加载文件完成...")

        for name in truth:
            self.name_count += 1
            truth_vk = dict()  # truth 里面每个文档及其对应的作者
            preds_vk = dict()  # pred 里面每个文档对应的簇 id
            for people_id, article_ids in truth[name].items():
                for article_id in article_ids:
                    art_id = article_id[:article_id.find('-')]
                    truth_vk[art_id] = people_id
            cluster_index = 0
            for article_ids in preds[name]:
                cluster_index += 1
                for article_id in article_ids:
                    art_id = article_id
                    # art_id = article_id[:article_id.find('-')]
                    preds_vk[art_id] = cluster_index
            precision_name, recall_name, f1_name = self.pairwise_count(truth_vk, preds_vk)

            self.precision += precision_name
            self.recall += recall_name
            self.f1 += f1_name
        return self.precision / self.name_count, self.recall / self.name_count, self.f1 / self.name_count, self.name_count

if __name__ == "__main__":
    v = Validation()
    truth_na_file = "name_to_pubs_train_500.json"
    preds_na_file = "result_aminder.json"
    precision, recall, f1, count = v.validate_na(truth_na_file, preds_na_file)
    print("准确率:  " + str(precision))
    print("召回率:  " + str(recall))
    print("f1 值:  " + str(f1))
    print("名称数量:  " + str(count))
