# encoding: utf-8


def combine_small_cluster(self):
    """
    对长度为2-4的cluster进行合并
    :return:
    """
    q = Queue.Queue(len(self.clusters))

    # 对当前簇按照从大到小顺序进行排序
    cluster_dict = {}
    for index, cluster in enumerate(self.clusters):
        cluster_dict[index] = [len(cluster.papers), cluster]
    sort_cluster_list = sorted(cluster_dict.items(), key=lambda x: x[1][0], reverse=True)
    sort_cluster_list = [cluster_pair[1][1] for cluster_pair in sort_cluster_list]

    # 只合并长度在1-3的cluster
    for cluster_item in sort_cluster_list:
        if 0 < len(cluster_item.papers) < 5:
            q.put(cluster_item)

    while not q.empty():
        current_cluster = q.get()
        not_merge_clusters = []

        while not q.empty():
            other_cluster = q.get()

            is_merge = False
            for current_paper in current_cluster.papers:
                for other_paper in other_cluster.papers:
                    if current_paper.venue == other_paper.venue and current_paper.venue != "" \
                            and (current_paper.org == "" or other_paper.org == ""):
                        self.merge_cluster(current_cluster.index, other_cluster.index)
                        is_merge = True
                        break

                if is_merge:
                    break
            if not is_merge:
                not_merge_clusters.append(other_cluster)

        for cluster_item in not_merge_clusters:
            q.put(cluster_item)

    def remove_noise(self):
        """
        去除当前每个簇中的噪音
        :return:
        """
        tmp_cluster_list = [cluster_item for cluster_item in self.clusters]

        for cluster_item in tmp_cluster_list:
            # 只对paper数量在8以上的cluster去除噪音
            if len(cluster_item.papers) > 8:

                cluster_paper_list = [paper for paper in cluster_item.papers]
                cluster_orgs = [paper.org for paper in cluster_paper_list if paper.org != ""]

                # 找出占比小于10%的org
                normal_org_set = set()
                abnormal_org_set = set()
                for org, org_num in Counter(cluster_orgs).items():
                    if float(org_num) / len(cluster_orgs) < 0.12:
                        abnormal_org_set.add(org)
                    elif float(org_num) / len(cluster_orgs) > 0.3:
                        normal_org_set.add(org)

                for paper in cluster_paper_list:
                    current_org = paper.org

                    doubt_count = 0
                    if current_org in abnormal_org_set:
                        for normal_org in normal_org_set:
                            if sent_distance(current_org, normal_org) > 20 \
                                    and sent_ratio(current_org, normal_org) < 0.3 \
                                    and (current_org not in normal_org) and (normal_org not in current_org):
                                    doubt_count += 1

                    # 当前org与簇中主要的org明显不同
                    if doubt_count == len(normal_org_set):
                        if current_org == cluster_item.main_org:
                            # 将当前paper从cluster中移除
                            self.remove_paper_from_cluster(cluster_item, paper)

                        elif sentence_sim(paper.title, cluster_item.main_title) < 0.5:
                            # 将当前paper从cluster中移除
                            self.remove_paper_from_cluster(cluster_item, paper)

    # def venue_run(self):
    #     """
    #     匹配veneu是否相等
    #     :return:
    #     """
    #     no_information_num = 0
    #     q = Queue.Queue(len(self.clusters))
    #     for cluster in self.clusters:
    #         q.put(cluster)
    #
    #     while not q.empty():
    #         main_cluster = q.get()
    #         not_merge_clusters = []
    #
    #         while not q.empty():
    #             cluster = q.get()
    #
    #             if main_cluster.main_venue == cluster.main_venue and self.cluster_co_athor(main_cluster, cluster) > 2:
    #                 self.merge_cluster(main_cluster.index, cluster.index)
    #             else:
    #                 not_merge_clusters.append(cluster)
    #         for cluster in not_merge_clusters:
    #             q.put(cluster)
    #     print("Number of no org information is:", no_information_num)

    @staticmethod
    def pair_cluster_static(current_cluster, other_cluster):
        """
        计算簇之间的相似度
        :param current_cluster:
        :param other_cluster:
        :return:
        """
        # 统计两个簇的相同关键词
        current_keyword_list = []
        for paper in current_cluster.papers:
            if len(paper.keywords) > 0:
                current_keyword_list.extend(paper.keywords)

        other_keyword_list = []
        for paper in other_cluster.papers:
            if len(paper.keywords) > 0:
                other_keyword_list.extend(paper.keywords)

        same_keyword_ratio = 0.0
        same_keyword_num = len(set(current_keyword_list) & set(other_keyword_list))
        if len(set(current_keyword_list) | set(other_keyword_list)) != 0:
            same_keyword_ratio = same_keyword_num * 1.0 / \
                                 len(set(current_keyword_list) | set(other_keyword_list))

        # 两个簇topK关键词的语义相似度
        cluster_keyword_similarity = 0.0
        current_top_keywords = []
        other_top_keywords = []
        if len(set(current_keyword_list)) > 3:
            current_top_keywords = [pair[0] for pair in
                                    sorted(Counter(current_keyword_list).items(),
                                           key=lambda x: x[1], reverse=True)[:3]]

        if len(set(other_keyword_list)) > 3:
            other_top_keywords = [pair[0] for pair in
                                  sorted(Counter(other_keyword_list).items(),
                                         key=lambda x: x[1], reverse=True)[:3]]

        current_keyword_vec = [vec_dict[word] for word in current_top_keywords if word in vocab_set]
        other_keyword_vec = [vec_dict[word] for word in other_top_keywords if word in vocab_set]
        if len(current_keyword_vec) > 0 and len(other_keyword_vec) > 0:
            cluster_keyword_similarity = 1 - spatial.distance.cosine(np.mean(current_keyword_vec, axis=0),
                                                                     np.mean(other_keyword_vec, axis=0))

        # 统计两个簇title之间的共有词
        current_title_word_list = []
        for paper in current_cluster.papers:
            if paper.title != "":
                current_title_word_list.append(paper.title)

        other_title_word_list = []
        for paper in other_cluster.papers:
            if paper.title != "":
                other_title_word_list.append(paper.title)

        same_title_word_ratio = 0.0
        if len(set(current_title_word_list) | set(other_title_word_list)) != 0:
            same_title_word_ratio = len(set(current_title_word_list) & set(other_title_word_list)) * 1.0 / \
                                    len(set(current_title_word_list) | set(other_title_word_list))

        # 统计两个簇的相同期刊
        current_venue_list = []
        for paper in current_cluster.papers:
            if paper.venue != "":
                current_venue_list.append(paper.venue)

        other_venue_list = []
        for paper in other_cluster.papers:
            if paper.venue != "":
                other_venue_list.append(paper.venue)

        same_venue_ratio = 0.0
        same_venue_num = len(set(current_venue_list) & set(other_venue_list))
        if len(set(current_venue_list) | set(other_venue_list)) != 0:
            same_venue_ratio = same_venue_num * 1.0 / \
                               len(set(current_venue_list) | set(other_venue_list))

        # 统计两个簇的相同机构
        current_org_list = []
        for paper in current_cluster.papers:
            if paper.org != "":
                current_org_list.append(paper.org)

        other_org_list = []
        for paper in other_cluster.papers:
            if paper.org != "":
                other_org_list.append(paper.org)

        same_org_ratio = 0.0
        same_org_num = 0

        # 对文章数多的簇遍历
        if len(current_cluster.papers) >= len(other_cluster.papers):
            other_org_set = set(other_org_list)
            for current_org in current_org_list:
                for other_org in other_org_set:
                    if sent_distance(current_org, other_org) < 3:
                        same_org_num += 1

        else:
            current_org_set = set(current_org_list)
            for other_org in other_org_list:
                for current_org in current_org_set:
                    if sent_distance(current_org, other_org) < 3:
                        same_org_num += 1

        if len(current_org_list) + len(other_org_list) != 0:
            same_org_ratio = float(same_org_num) / \
                             (len(current_org_list) + len(other_org_list))

        # 两个簇主paper之间的相似性
        main_paper_feas = similarity(current_cluster.main_paper_feasain_paper, other_cluster.main_paper)

        return [same_keyword_ratio, same_title_word_ratio, same_venue_ratio, same_org_ratio, same_keyword_num, \
                same_venue_num, same_org_num, cluster_keyword_similarity, main_paper_feas]
