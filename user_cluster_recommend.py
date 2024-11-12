import numpy as np
import math
import time
import random
from dataset_processing import sampling
from options import args_parser
from collections import Counter
from itertools import chain
from data_set import convert
from utils import count_top_items
from sklearn.metrics.pairwise import cosine_similarity

def get_user_cluster(user_id, user_movie, user_movie_cos):
    """

    :param user_id: 当前用户id（测试集里面的用户id）
    :param user_movie: user-movie-rating矩阵（还包含用户信息）, 这个用户可不是测试用户的信息
    :param user_movie_cos: user-movie-rating矩阵压缩后： 12 * 103
    :return:
    """
    # 计算余弦相似度
    user_prob = np.zeros((user_movie.shape[0], 2)) # 用户之间的余弦值 ： 用户数 * 2
    user_prob[:, 0] = user_movie[:, 0]             # 第一列为用户id
    # print(user_movie[:, 0] == user_id) ：[ True False False False False False False False False False False]
    # user_movie_cos[user_movie[:, 0] == user_id] 获取user_movie[:, 0] == user_id那一用户的数据
    user_prob[:, 1] = cosine_similarity(user_movie_cos, user_movie_cos[user_movie[:, 0] == user_id]).T # # 用当前用户的编码器输出和其他用户的编码器输出的余弦距离
    # np.lexsort 多序列排序，加－号是为了从大到小排序
    user_prob = user_prob[np.lexsort(-user_prob.T), :]
    # 选取最大的10个作为user_sim相似用户, 如果没有10个就选全部的
    user_sim = user_prob[0:10, 0] if user_prob.shape[0] >= 10 else user_prob[0:, 0]
    user_sim = user_sim.astype(np.uint16)
    return user_sim


def get_user_recommend(user_sim, test_dataset):
    # 遍历相似用户,找到他们的观看历史
    # user_id  测试用户
    # user_sim 和 user_id 相似的10个用户, test_dataset 测试用户集
    watch_history = []
    for i in user_sim:
        # test_dataset[:, 1] 是 movie_id
        movies_in_user = test_dataset[:, 1][test_dataset[:, 0] == i].astype(np.uint16)
        # 和 user_id相似的用户看过的电影
        watch_history.append(list(movies_in_user))
    finallist = count_top_items(10, watch_history) # 找到观看历史中次数最多的前num部电影
    return finallist


def recommend(user_movie, test_dataset, weights):
    """
    :param user_movie: user-movie-rating 第一列用户id，1682个电影，最后三列为用户信息（和test_dataset最后三列信息一样），12个用户
    :param weights: model
    :param test_dataset: 第i个client的测试集矩阵 14个client
           ，user_num*6（['user_id', 'movie_id', 'rating', 'gender', 'age', 'occupation']，
    :return:recommend_list: 电影的推荐列表
    user_movie 里面的user_id 和 test_dataset 里面的user_id是一样的，只不过test_dataset里面一个用户可能看了多个电影
    """
    # 计算user_movie_cos
    # user_movie_cos 是user_movie经过压缩得到的
    weights = np.array(weights)
    user_movie_cos = np.zeros((user_movie.shape[0], weights.shape[0] + 3))  # 用户数 * 100
    user_movie_cos[:, [-3, -2, -1]] = user_movie[:, [-3, -2, -1]]           # 用户数 * 103
    # print('np.dot')
    # print(time.time())
    "user_movie[:, 1:-3](用户数*1682): 中间1682个电影的评分"
    "weights.T : 1682 * 100"
    "np.dot(user_movie[:, 1:-3], weights.T)"
    user_movie_cos[:, 0:-3] = np.dot(user_movie[:, 1:-3], weights.T)        # 用户数 * 103

    # 得到第i个用户的测试集中的电影
    user_in_dataset = test_dataset[:, 0].astype(np.uint16) # 第一列：用户id，用户可以多次评分
    count = Counter(user_in_dataset)
    # all_list: 用来保存一个client中每个用户所得到推荐列表的总和
    all_list = []
    # 遍历user id
    # 设置活跃用户为访问最多的1/3
    request_times = sorted(count.values())  # 用户活跃度排序
    # 列表的长度就代表了所有用户活跃度的总和
    active_times = request_times[int(len(request_times)/3*2)]
    for (user_id, times) in count.items():
        # 遍历测试集中的用户，得到他们的相似用户
        # 活跃用户应该如何界定，是items多的用户吗，这个多应该如何界定
        # 设置1/3的活跃用户
        # 判断该用户的活跃度是否大于active活跃度
        if times > active_times:
            user_sim = get_user_cluster(user_id, user_movie, user_movie_cos)
            # 得到用户的推荐10人，以及他们的推荐列表中最多的10个
            finallist = get_user_recommend(user_sim, test_dataset)
            all_list.append(list(finallist))
    return all_list


def Oracle_recommend(test_dataset, cachesize):
    """
       返回一个client最终的推荐列表：选择test_dataset中出现最多的cachesize部电影作为推荐列表，
       :param test_dataset: 第i个client的测试集矩阵
               ，user_num*6（['user_id', 'movie_id', 'rating', 'gender', 'age', 'occupation']，
       :param cachesize: 缓存电影的数目
       :return:recommend_list: 电影的推荐列表
       """
    # 得到test_dataset中所有请求的movie_id
    movie_in_dataset = test_dataset[:, 1].astype(np.uint16)
    count = Counter(movie_in_dataset)
    recommend_list = np.array(count.most_common(cachesize))[:, 0]
    return recommend_list


def Greedy_recommend(test_dataset, cachesize, e, movie_id_max):
    """
    返回一个client最终的推荐列表：以1-e的概率选择test_dataset中出现最多的cachesize部电影作为推荐列表，
    以e的概率随机选择cachesize部电影作为推荐列表。
    :param test_dataset: 第i个client的测试集矩阵
               ，user_num*6（['user_id', 'movie_id', 'rating', 'gender', 'age', 'occupation']，
    :param cachesize:缓存电影的数目
    :param e: algorithm  parameters
    :param movie_id_max: 电影的最大索引，最后一部电影
    :return:recommend_list: 电影的推荐列表
    """

    if random.random() <= e:
        recommend_list = np.random.choice(range(1, movie_id_max + 1), cachesize, replace=False)
    else:
        recommend_list = Oracle_recommend(test_dataset, cachesize)

    return recommend_list


if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    sample, users_group_train, users_group_test = sampling(args)
    # 设置weights为全1
    weights = np.ones((100, max(sample['movie_id'])))
    # 以client 0为例子，即使用users_group_train[0] 、users_group_test[0]作为idx
    client_0 = np.array(sample.iloc[users_group_test[0], :])
    user_movie_0 = convert(client_0, max(sample['movie_id']))
    print('convert over\n')
    recommend_movies = recommend(user_movie_0, client_0, weights)
    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
