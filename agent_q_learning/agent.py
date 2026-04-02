#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)
from kaiwu_agent.utils.common_func import create_cls, attached
from kaiwu_agent.agent.base_agent import BaseAgent
from agent_q_learning.conf.conf import Config
from agent_q_learning.algorithm.algorithm import Algorithm
from agent_q_learning.feature.definition import ObsData


#ObsData = create_cls("ObsData", feature=None)
ActData = create_cls("ActData", act=None)


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        self.logger = logger

        # Initialize parameters
        # 参数初始化
        self.state_size = Config.STATE_SIZE
        self.action_size = Config.ACTION_SIZE
        self.learning_rate = Config.LEARNING_RATE
        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON
        self.episodes = Config.EPISODES
        self.algorithm = Algorithm(self.gamma, self.learning_rate, self.state_size, self.action_size)
        self.current_target_id = None

        super().__init__(agent_type, device, logger, monitor)

    @predict_wrapper
    def predict(self, list_obs_data):
        """
        The input is list_obs_data, and the output is list_act_data.
        """
        """
        输入是 list_obs_data, 输出是 list_act_data
        """
        state = list_obs_data[0].feature
        act = self._epsilon_greedy(state=state, epsilon=self.epsilon)

        return [ActData(act=act)]

    @exploit_wrapper
    def exploit(self, list_obs_data):
        state = list_obs_data[0].feature
        act = np.argmax(self.algorithm.Q[state, :])

        return [ActData(act=act)]

    def _epsilon_greedy(self, state, epsilon=0.1):
        """
        Epsilon-greedy algorithm for action selection
        """
        """
        ε-贪心算法用于动作选择
        """
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, self.action_size)

        # Exploitation
        # 探索
        else:
            """
            Break ties randomly
            If all actions are the same for this state we choose a random one
            (otherwise `np.argmax()` would always take the first one)
            """
            """
            随机打破平局,在某些情况下，当有多个动作或策略具有相同的评估值或优先级时，需要进行决策。
            为了避免总是选择第一个动作或策略，可以使用随机选择的方法来打破平局。以增加多样性和随机性
            """
            if np.all(self.algorithm.Q[state, :]) == self.algorithm.Q[state, 0]:
                action = np.random.randint(0, self.action_size)
            else:
                action = np.argmax(self.algorithm.Q[state, :])

        return action

    @learn_wrapper
    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def observation_process(self, raw_obs, game_info):
        # 1. 基础信息：当前坐标
        pos_x, pos_z = game_info.pos_x, game_info.pos_z
        pos_id = int(pos_x * 64 + pos_z)
        
        # 2. 宝箱状态：0表示已消失，1表示存在
        treasure_status = [int(x if x != 2 else 0) for x in game_info.treasure_status]
        
        # 3. 距离信息：raw_obs[0]是到终点距离，raw_obs[1:11]是到10个固定点的距离
        dist_to_end = raw_obs[0]
        dist_to_fixed_points = raw_obs[1:11]

        # 4. 确定当前目标固定点索引 (target_id)
        # 如果当前没有目标，或者原目标宝箱已经消失，则重新寻找最近的宝箱
        if self.current_target_id is None or self.current_target_id == 0 or treasure_status[self.current_target_id - 1] == 0:
            new_target = 0 # 默认去终点
            min_dist = 999.0
            for i in range(10):
                if treasure_status[i] == 1:
                    if dist_to_fixed_points[i] < min_dist:
                        min_dist = dist_to_fixed_points[i]
                        new_target = i + 1
            self.current_target_id = new_target

        # 5. 组合 State (去耦合设计)
        # 空间大小：4096 * 11 = 45056
        # 这个状态只取决于：我在哪 + 我现在要去哪个固定点
        state = pos_id * 11 + self.current_target_id

        # 提取局部视野用于奖励计算中的碰撞检测
        local_view = [game_info.local_view[i : i + 5] for i in range(0, len(game_info.local_view), 5)]
        obstacle_flat = []
        for sub_list in local_view:
            obstacle_flat.extend([1 if i == 0 else 0 for i in sub_list])

        return ObsData(
            feature=int(state),
            raw_obs=raw_obs,
            target_id=self.current_target_id, # 传给奖励函数
            pos=[pos_x, pos_z],
            local_obstacle=obstacle_flat,
            treasure_status=treasure_status,
            memory=game_info.location_memory # 用于路径重复惩罚
        )

    def action_process(self, act_data):
        return act_data.act

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
        np.save(model_file_path, self.algorithm.Q)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
        try:
            self.algorithm.Q = np.load(model_file_path)
            self.logger.info(f"load model {model_file_path} successfully")
        except FileNotFoundError:
            self.logger.info(f"File {model_file_path} not found")
            exit(1)
