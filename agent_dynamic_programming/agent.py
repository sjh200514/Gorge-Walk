#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import copy
from kaiwu_agent.utils.common_func import create_cls, attached
import numpy as np
from kaiwu_agent.agent.base_agent import BaseAgent
from kaiwu_agent.agent.base_agent import (
    save_model_wrapper,
    learn_wrapper,
    load_model_wrapper,
)
from agent_dynamic_programming.conf.conf import Config
from agent_dynamic_programming.algorithm.algorithm import Algorithm

ObsData = create_cls("ObsData", feature=None)
ActData = create_cls("ActData", act=None)


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        self.logger = logger
        
        # 【优化后的顺序】：按照空间连续性排列，防止大幅度折返导致卡死
        self.target_sequence = [1230, 620, 604, 2071, 3497, 3192, 2657, 2733, 2298, 1527, 759]
        self.current_stage = 0 

        # 状态空间为 4096 * 11
        self.algorithm = Algorithm(
            Config.GAMMA, Config.THETA, Config.EPISODES, 45056, Config.ACTION_SIZE, self.logger
        )

        super().__init__(agent_type, device, logger, monitor)

    def predict(self, state):
        return np.argmax(self.algorithm.agent_policy[state])

    def exploit(self, state):
        return np.argmax(self.algorithm.agent_policy[state])

    @learn_wrapper
    def learn(self, F):
        self.algorithm.learn(F)

    def observation_process(self, raw_obs, game_info):
        pos_id = int(game_info.pos_x * 64 + game_info.pos_z)

        # 判定是否到达当前阶段目标
        if pos_id == self.target_sequence[self.current_stage]:
            if self.current_stage < len(self.target_sequence) - 1:
                self.current_stage += 1

        # 构造复合状态
        combined_state = pos_id * 11 + self.current_stage
        return ObsData(feature=int(combined_state))

    def action_process(self, act_data):
        pass

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
        np.save(model_file_path, self.algorithm.agent_policy)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
        try:
            self.algorithm.agent_policy = np.load(model_file_path)
            self.current_stage = 0 
            self.logger.info(f"load model {model_file_path} successfully")
        except FileNotFoundError:
            exit(1)