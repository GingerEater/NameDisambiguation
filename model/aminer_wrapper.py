#!/usr/bin/python3
# coding: utf-8
import json
import file_config

from cluster_model import Person

if __name__ == '__main__':
    aminer_data = json.load(open(file_config.external_v2_data))
    aminer_ans_data = json.load(open(file_config.external_v2_label))
    print(len(aminer_ans_data.keys()))  # 500
    print(list(aminer_ans_data.keys())[:3])  # ['yanjun_zhang', 'lijie_qiao', 'k_yang']
    print(list(aminer_data.keys())[0])  # 5b5433eae1cd8e4e15039b23

    # ##################################################################
    # ## deal with aminer_ans_data
    raw_data = {}
    for name in aminer_ans_data.keys():
        # for entity in aminer_ans_data[name].keys():
            # paper_ids = aminer_ans_data[name][entity]
        paper_ids_list = [aminer_ans_data[name][entity] for entity in aminer_ans_data[name].keys()]
        raw_data[name] = [paper_id.split('-')[0] for paper_ids in paper_ids_list for paper_id in paper_ids]
    print(len(raw_data['yanjun_zhang']))  # 462
    print(len(raw_data['lijie_qiao']))  # 217
    print(raw_data['lijie_qiao'][:10])  #

    # ##################################################################
    # ## deal raw data to dict
    paper_dict = {}
    for paper_id in aminer_data.keys():
        aminer_data[paper_id]['id'] = paper_id
        paper_dict[paper_id] = aminer_data[paper_id]

    # ##################################################################
    # ## map the result
    for name in raw_data.keys():
        raw_data[name] = [paper_dict[paper_id] for paper_id in raw_data[name]]
    json.dump(raw_data, open('tmp_aminer.data', 'w'), indent=4)

    ## 上面是对 v2 数据进行转换
    ## 下面写入文件
    aminder_process_data = json.load(open('./tmp_aminer.data'))
    res = dict()

    for i, name in enumerate(aminder_process_data.keys()):
        print(i)
        person = Person(' '.join(name.split('_')), aminder_process_data[name])
        person.co_author_run()
        person.org_run()
        res[name] = [cluster.output() for cluster in person.clusters]
    json.dump(res, open('result_aminder.json', 'w'), indent=4)
    print(res['yanjun_zhang'])
