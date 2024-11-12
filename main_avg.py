import copy
import time
import numpy as np
from tqdm import tqdm
import torch
from itertools import chain
import matplotlib.pyplot as plt
from scipy import stats

from options import args_parser
from dataset_processing import sampling, average_weights,asy_average_weights, sampling_mobility
from user_cluster_recommend import recommend
from local_update import LocalUpdate, cache_hit_ratio
from model import AutoEncoder
from utils import exp_details, ModelManager, count_top_items
from data_set import convert
from select_vehicle import select_vehicle, vehicle_p_v, select_vehicle_mobility, vehicle_p_v_mobility
from cv2x import V2Ichannels, Environ
from dueling_ddqn import DuelingAgent, mini_batch_train
from environment import CacheEnv


if __name__ == '__main__':
    v2i_rate_all = []
    v2i_rate_mbs_all = []
    idx = 0
    # 开始时间
    start_time = time.time()
    # args & 输出实验参数
    args = args_parser()
    exp_details(args)
    # gpu or cpu
    if args.gpu: torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load sample users_group_train users_group_test
    sample, users_group_train, users_group_test, request_content, vehicle_request_num = sampling_mobility(args,
                                                                                                          args.clients_num)
    """
    1.	sample:一个包含用户和电影信息的 pandas DataFrame。
	2.	users_group_train:一个字典,键为车辆编号,值为训练数据的索引列表。
	3.	users_group_test:一个字典,键为车辆编号,值为测试数据的索引列表。
	4.	request_content:一个字典,键为训练轮次epoch编号 值为请求内容的索引列表。
	5.	vehicle_request_num:一个字典 键为训练轮次编号 值为每个车辆在该轮次的请求数量列表。
        sample: pandas DataFrame,包含用户和电影信息
            # 示例：
            # sample 的内容如下：
            # Index  user_id  movie_id  rating  gender  age  occupation
            # 0      1        101       5.0     0       25   1
            # 1      1        102       4.0     0       25   1
            # 2      2        101       3.0     1       30   2
            # 3      2        103       4.5     1       30   2
            # 4      3        102       2.0     0       22   3
            # 5      3        104       5.0     0       22   3
            # 访问方法：
            # sample.iloc[i]  # 获取第 i 行的数据（行索引从 0 开始）
            # sample.iloc[i]['user_id']  # 获取第 i 行的 'user_id'

        users_group_train: dict,键为车辆编号,值为训练数据的索引列表
            # 示例：
            # users_group_train = {
            #     0: [0, 1],     # 车辆 0 的训练数据索引列表
            #     1: [2, 3]      # 车辆 1 的训练数据索引列表
            # }
            # 访问方法：
            # train_indices_vehicle_0 = users_group_train[0]  # 获取车辆 0 的训练数据索引列表
            # index = train_indices_vehicle_0[0]  # 获取车辆 0 的第一个训练样本索引
            # sample_data = sample.iloc[index]  # 获取对应的样本数据

        users_group_test: dict,键为车辆编号,值为测试数据的索引列表
            # 示例：
            # users_group_test = {
            #     0: [],          # 车辆 0 的测试数据索引列表（为空）
            #     1: [4, 5]      # 车辆 1 的测试数据索引列表
            # }
            # 访问方法与 users_group_train 类似

        request_content: dict,键为训练轮次(epoch)编号,值为请求内容的索引列表
            # 示例：
            # request_content = {
            #     0: [4],        # 第 0 轮训练的请求内容索引列表
            #     1: [5]         # 第 1 轮训练的请求内容索引列表
            # }
            # 访问方法：
            # request_indices_epoch_0 = request_content[0]  # 获取第 0 轮的请求内容索引列表
            # sample_data = sample.iloc[request_indices_epoch_0[0]]  # 获取请求的样本数据

        vehicle_request_num: dict,键为训练轮次编号,值为每个车辆在该轮次的请求数量列表
            # 示例：
            # vehicle_request_num = {
            #     0: [0, 1],     # 第 0 轮中，每个车辆的请求数量（车辆 0 请求 0 个，车辆 1 请求 1 个）
            #     1: [0, 1]      # 第 1 轮中，每个车辆的请求数量
            # }
            # 访问方法：
            # request_num_epoch_0 = vehicle_request_num[0]  # 获取第 0 轮的车辆请求数量列表
            # num_requests_vehicle_1_epoch_0 = request_num_epoch_0[1]  # 获取车辆 1 在第 0 轮的请求数量
    """

    print('different epoch vehicle request num', vehicle_request_num)

    data_set = np.array(sample)
    """
    data_set = np.array([
    [1, 101, 5.0, 0, 25, 1],  # Index 0
    [1, 102, 4.0, 0, 25, 1],  # Index 1
    [2, 101, 3.0, 1, 30, 2],  # Index 2
    [2, 103, 4.5, 1, 30, 2],  # Index 3
    [3, 102, 2.0, 0, 22, 3],  # Index 4
    [3, 104, 5.0, 0, 22, 3],  # Index 5
    ])
    """
    # test_dataset & test_dataset_idx
    test_dataset_idxs = []
    for i in range(args.clients_num): # 这里假设两个clients 
        test_dataset_idxs.append(users_group_test[i]) # test_dataset_idxs = [[], [4, 5]]
    test_dataset_idxs = list(chain.from_iterable(test_dataset_idxs)) # 结果：test_dataset_idxs = [4, 5]
    test_dataset = data_set[test_dataset_idxs] # 即：test_dataset = data_set[[4, 5]]  
    """
    test_dataset = np.array([
    [3, 102, 2.0, 0, 22, 3],  # Index 4
    [3, 104, 5.0, 0, 22, 3],  # Index 5
])
    """

    request_dataset = []

    # 原代码
    # for i in range(args.epochs): 
    #     request_dataset_idxs = []
    #     request_dataset_idxs.append(request_content[i]) 
    #     request_dataset_idxs = list(chain.from_iterable(request_dataset_idxs)) #
    #     request_dataset.append(data_set[request_dataset_idxs])
    for i in range(args.epochs): 
        request_dataset_idxs = request_content[i]  # 不要初始化为空，直接获取当轮的请求内容
        request_dataset.append(data_set[request_dataset_idxs])  # 添加当前轮的数据
        """data_set[4] = [3, 102, 2.0, 0, 22, 3] 
        request_dataset[0] = np.array([[3, 102, 2.0, 0, 22, 3]])
        data_set[5] = [3, 104, 5.0, 0, 22, 3]
        request_dataset[1] = np.array([[3, 104, 5.0, 0, 22, 3]])
        """

    all_pos_weight, veh_speed, veh_dis = select_vehicle_mobility(args.clients_num)
    """
    •	返回三个变量：
	•	all_pos_weight: 每辆车的单位位置权重,均为1。
	•	veh_speed: 每辆车的速度数组,单位为m/s。
        截断的正态分布随机采样(50-60)*0.278 m/s
	•	vehicle_dis: 每辆车距离初始位置的距离数组(当前为0)。
    """
    time_slow = 0.1

    # c-v2x simulation parameters: 车载蜂窝网
    V2I_min = 100  # minimum required data rate for V2I Communication 汽车和rsu的通信 它规定了车辆与路侧单元（RSU）或基站之间的通信速率下限，确保车联网应用能够获得足够的数据传输速率。
    bandwidth = int(540000) #车辆与路侧单元（RSU，Road-Side Unit）之间的通信带宽。
    bandwidth_mbs = int(1000000) #车辆与蜂窝基站（MBS，Mobile Base Station）之间的通信带宽。

    env = Environ(args.clients_num, V2I_min, bandwidth, bandwidth_mbs) 

    env.new_random_game(veh_dis, veh_speed)  # initialize parameters in env
    """
    主要是计算了慢衰落(路径损失和阴影衰落） 以及快衰落 短时间内的多径效应。
    """

    # build model
    global_model = AutoEncoder(int(max(data_set[:, 1])), 100)
    V2Ichannels = V2Ichannels()

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()

    vehicle_model_dict = [[], [], [], [], [], [], [], [], [], [], [], [], []
        , [], [], [], [], [], [], []]
    for i in range(args.clients_num):
        vehicle_model_dict[i].append(copy.deepcopy(global_model))
    # copy weights
    global_weights = global_model.state_dict() #主要是采用正态分布或者均匀分布 权重初始化

    # all epoch weights
    w_all_epochs = dict([(k, []) for k in range(args.epochs)])

    # Training loss
    train_loss = []

    # each epoch train time
    each_epoch_time=[]
    each_epoch_time.append(0)

    vehicle_leaving=[]

    v2i_rate_epoch=dict([(k, []) for k in range(args.epochs)])
    v2i_rate_mbs_epoch = dict([(k, []) for k in range(args.epochs)])

    cache_efficiency_list=[]
    cache_efficiency_without_list=[]

    request_delay_list=[]

    while idx < args.epochs:
        # 开始
        print(f'\n | Global Training Round : {idx + 1} |\n')
        global_model.train()
        
        # 原始代码（注释）
        # local_net = copy.deepcopy(vehicle_model_dict[idx % args.clients_num][-1])
        # local_net.to(device)
        
        # v2i rate
        v2i_rate, v2i_rate_mbs = env.Compute_Performance_Train_mobility(args.clients_num)
        v2i_rate_mbs_all.append(v2i_rate_mbs)
        v2i_rate_all.append(v2i_rate)
        print('v2i rate', v2i_rate)
        print('v2i rate mbs', v2i_rate_mbs)
        v2i_rate_epoch[idx] = v2i_rate
        v2i_rate_mbs_epoch[idx] = v2i_rate_mbs
        v2i_rate_weight = v2i_rate / max(v2i_rate)
        print('vehicle position', veh_dis)
        print('vehicle speed', veh_speed)
        
        # 原始代码（注释）
        # print("vehicle ", idx % args.clients_num + 1, " start synchronous training for ", args.local_ep)
        # epoch_start_time = time.time()
        # local_weights_avg=[]
        # for veh in range(15):
        #     local_model = LocalUpdate(args=args, dataset=data_set,
        #                               idxs=users_group_train[idx % args.clients_num])
        #     w, loss, local_net = local_model.update_weights(
        #         model=local_net, client_idx=idx % args.clients_num + 1, global_round=idx + 1)
        #     local_weights_avg.append(copy.deepcopy(w))
        
        # 修改后的代码
        epoch_start_time = time.time()
        local_weights = []
    
        for veh in range(args.clients_num):  # 遍历所有车辆
            print(f"Vehicle {veh + 1} start synchronous training for {args.local_ep} epochs")
            
            local_net = copy.deepcopy(vehicle_model_dict[veh][-1])  # 获取该车辆的最新模型
            local_net.to(device)
            
            local_model = LocalUpdate(args=args, dataset=data_set,
                                    idxs=users_group_train[veh])  # 使用该车辆的本地数据
            w, loss, updated_local_net = local_model.update_weights(
                model=local_net, client_idx=veh + 1, global_round=idx + 1)
            
            vehicle_model_dict[veh].append(updated_local_net)  # 保存更新后的模型
            local_weights.append(w)
    
        # 原始代码（注释）
        # # update global weights
        # global_weights_avg = average_weights(local_weights_avg)
        # # update global weights
        # global_model.load_state_dict(global_weights_avg)
        
        # 修改后的代码
        # update global weights
        global_weights_avg = average_weights(local_weights)
        # update global weights
        global_model.load_state_dict(global_weights_avg)
        
        epoch_time = time.time() - epoch_start_time
        each_epoch_time.append(epoch_time)
        
        # 原始代码（注释）
        # w_all_epochs[idx] = global_weights_avg['linear1.weight'].tolist()
        
        # 修改后的代码
        w_all_epochs[idx] = global_weights['linear1.weight'].tolist()

        ##DDQN
        cache_size=100
        MAX_EPISODES = 30
        MAX_STEPS = 200
        BATCH_SIZE = 32
        recommend_movies_c500 = []
        for i in range(args.clients_num):
            vehicle_seq = i
            test_dataset_i = data_set[users_group_test[vehicle_seq]]
            user_movie_i = convert(test_dataset_i, max(sample['movie_id']))
            recommend_list = recommend(user_movie_i, test_dataset_i, w_all_epochs[idx])
            recommend_list500 = count_top_items(int(2.5 * cache_size), recommend_list)
            recommend_movies_c500.append(list(recommend_list500))

        # AFPCC
        recommend_movies_c500 = count_top_items(int(2.5 * cache_size), recommend_movies_c500)
        # recommend_movies_c500 存放了推荐的电影；cache_size = 100
        env_rl = CacheEnv(recommend_movies_c500,cache_size)
        agent = DuelingAgent(env_rl,cache_size)
        episode_rewards, cache_efficiency, request_delay = mini_batch_train(env_rl, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE,
                                           request_dataset[idx], v2i_rate,v2i_rate_mbs,
                                        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], vehicle_request_num[idx])

        cache_efficiency_list.append(cache_efficiency[-1])
        cache_efficiency_without_list.append(cache_efficiency[0])

        request_delay_list.append(request_delay[-1])

        idx += 1
        veh_dis, veh_speed ,all_pos_weight = vehicle_p_v_mobility(veh_dis , epoch_time, args.clients_num, idx, args.clients_num)


        env.renew_channel(args.clients_num, veh_dis, veh_speed)  # update channel slow fading
        env.renew_channels_fastfading()  # update channel fast fading

        if idx == args.epochs:

            cache_efficiency_list.insert(0, 0)
            for i in range(len(cache_efficiency_list)):
                cache_efficiency_list[i] *= 100
            print('Cache hit radio',cache_efficiency_list)

            cache_efficiency_without_list.insert(0, 0)
            for i in range(len(cache_efficiency_without_list)):
                cache_efficiency_without_list[i] *= 100
            print('Cache hit radio without RL',cache_efficiency_without_list)


            print('each_epoch_time',each_epoch_time)
            print('request_delay',request_delay_list)


        if idx > args.epochs:
            break

