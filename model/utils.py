#!/usr/bin/python3
# coding: utf-8
import Levenshtein
import time
from datetime import timedelta
import file_config
import re

with open(file_config.name_differ_test_path) as f:
    similarity_words = set([word for line in f for word in line.split()])

def get_time_dif(start_time):
    """
    获取已使用时间
    :param start_time: 起始时间
    :return:
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def org_process(org):
    global similarity_words
    org = org.strip().lower()

    ## 符号替换, 末尾 s 去掉
    org = ' '.join(org.split('-'))
    org = org.replace('，', ',') \
        .replace('.', '') \
        .replace(',', '') \
        .replace('’', "'") \
        .replace('(', '') \
        .replace(')', '') \
        .replace('/', ' ')
    org = ' '.join([item[:-1] if item[-1] == 's' else item for item in org.split()])

    ## 常用词简写替换
    org = ' ' + org + ' '
    org =  org.replace(' school ', ' college ') \
        .replace(' collage ', ' college ') \
        .replace(' colleage ', ' college ') \
        .replace(' a andf ', ' a f ') \
        .replace(' aandf ', ' a f ') \
        .replace(' a&f ', ' a f ') \
        .replace(' acad ', ' academy ') \
        .replace(' academic ', ' academy ') \
        .replace(' burning ', ' burn  ') \
        .replace(' ji lin ', ' jilin ') \
        .replace(' chinese academy science ca ', ' chinese academy science ') \
        .replace(' designing ', ' design ') \
        .replace(' depatment ', ' department ') \
        .replace(' dept ', ' department ') \
        .replace(' univ ', ' university ') \
        .replace(' sci ', ' science ') \
        .replace(' scien ', ' science ') \
        .replace(' phy ', ' physic ') \
        .replace(' physic ', ' physical ') \
        .replace(' prov ', ' province ') \
        .replace(' coll ', ' college ') \
        .replace(' syst ', ' system ') \
        .replace(' provincial ', ' province ') \
        .replace(' pharmacological ', ' pharmaceutical ') \
        .replace(' pharmaceutical ', ' pharmaceut ') \
        .replace(' tech ', ' technol ') \
        .replace(' technique ', ' technol ') \
        .replace(' technol ', ' technology ') \
        .replace(' technologie ', ' technology ') \
        .replace(' technical ', ' technological ') \
        .replace('ology ', 'ological ') \
        .replace(' res ', ' research') \
        .replace(' lab ', ' laboratory ') \
        .replace(' inst ', ' institute ') \
        .replace(' institute ', ' institution ') \
        .replace(' indust ', ' industrial ') \
        .replace(' surgery ', ' surgical ') \
        .replace(' metall ', ' metallurgy ') \
        .replace(' metallurgy ', ' metallurgical ') \
        .replace(' medicinal ', ' medical ') \
        .replace(' medicine ', ' medical ') \
        .replace(' mater ', ' material ') \
        .replace(' geographic ', ' geographical ') \
        .replace(' economy ', ' economic ') \
        .replace(' elec ', ' electric ') \
        .replace(' electrical ', ' electronic ') \
        .replace(' electric ', ' electronic ') \
        .replace(' electricity ', ' electronic ') \
        .replace(' catalysis ', ' catalytical ') \
        .replace(' comp ', ' computer ') \
        .replace(' computing ', ' computer ') \
        .replace(' environ ', ' environment ') \
        .replace(' environment ', ' environmental ') \
        .replace(' botany ', ' botanical ') \
        .replace(' pollutant ', ' pollution ') \
        .replace(' measuring ', ' measurement ') \
        .replace(' astr ', ' astronic ') \
        .replace(' astronic ', ' astronautics ') \
        .replace(' dev ', ' develop ') \
        .replace(' develop ', ' development ') \
        .replace(' developing ', ' development ') \
        .replace(' china ', ' chinese ') \
        .replace(' mine ', ' mineral ') \
        .replace(' northwestern ', ' northwest ') \
        .replace(' instrumentation ', ' instrument ') \
        .replace(' instrum ', ' instrument ') \
        .replace(' chem ', ' chemistry ') \
        .replace(' chemicals ', ' chemical ') \
        .replace(' chemical ', ' chemistry ') \
        .replace(' pediatric ', ' paediatric ') \
        .replace(' mech ', ' mechanic ') \
        .replace(' eng ', ' engineer ') \
        .replace(' cent ', ' center ') \
        .replace(' centre ', ' center ') \
        .replace(' ctr ', ' center ') \
        .replace(' stt ', ' state ') \
        .replace(' natl ', ' national ') \
        .replace(' state ', ' national ') \
        .replace(' engineer ', ' engineering ') \
        .replace(' electron ', ' electronic ') \
        .replace(' mechatronic ', ' mechatronical ') \
        .replace('cincinnati oh', 'cincinnati') \
        .replace(' compo ', ' composite ') \
        .replace(' and ', ' ') \
        .replace(' & ', ' ') \
        .replace(' the ', ' ') \
        .replace(' in ', ' ') \
        .replace(' for ', ' ') \
        .replace(' of ', ' ') \
        .replace(' at ', ' ') \
        .replace(' to ', ' ') \
        .replace(' on ', ' ') \
        .replace(' co ', ' ') \
        .replace(' college ', ' ') \
        .replace(' department ', ' ')

    ## 替换数字短语
    org = re.sub(r'[0-9]+th', '', org)
    org = re.sub(r'[0-9]', '', org)

    org = org.strip()


    ## 处理相近词
    for word in similarity_words:
        org = (' ' + org + ' ').replace(' ' + word + ' ', ' ' + word + word + word + word + ' ').strip()

    # if len(org) < 15:
    #     org = org * 5
    return org

def sen_distance(sen_first, sen_second):
    sen_dist = 100
    if sen_first != "" and sen_second != "":
        sen_dist = Levenshtein.distance(sen_first.strip(), sen_second.strip())

    return sen_dist


def is_org_contain_keyword(org):
    is_contain = False

    org_word_list = org.split()
    if "university" in org_word_list and len(org_word_list) > 1:
        is_contain = True

    return is_contain

def is_org_special(org):
    is_special = False

    org_list = ["graduate chinese academy science", "university chinese academy science"]
    if org in set(org_list):
        is_special = True

    return is_special


if __name__ == '__main__':
    org = 'mech hainan univ'; print(org_process(org))  # mechanics hainan university
    print(org_process('Academy of Armored Force Engineering'))  # academy armored force engineering
    # org = 'Academy of Armored Force Engineering'
    print(org_process('Advanced Materials and Technologies Institute'))  # advanced material technological institution
    org = 'Advanced Materials and Technologies Institute'
    print(org_process(' Institutes, Institute of '))  # institution institute; 第二个无法识别, 因为带空格有冲突
    print(' Institutes, Institute of '.lower().replace('institutes', 'institute').replace('institute', 'institution'))  # 以为这个前后没有空格
    print(org_process('Optoelectronics-Huazhong'))  # optoelectronic huazhong
    org = 'dept computer sci/engineering'; print(org_process(org))  # dept computer science engineering
    org = 'department otorhinolaryngology/head'; print(org_process(org))  # department otorhinolaryngological head
    org = 'college chem/chem engineering'; print(org_process(org))  # college chemistry chem engineering
    org = '309th hospital pla'; print(org_process(org))  # hospital pla
