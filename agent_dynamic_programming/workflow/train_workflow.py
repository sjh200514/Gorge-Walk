#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from kaiwu_agent.utils.common_func import attached
import time
import os
from tools.map_data_utils import read_map_data
from tools.train_env_conf_validate import check_usr_conf, read_usr_conf
from tools.metrics_utils import get_training_metrics


@attached
def workflow(envs, agents, logger=None, monitor=None):
    usr_conf = read_usr_conf("agent_dynamic_programming/conf/train_env_conf.toml", logger)
    if usr_conf is None: return
    if not check_usr_conf(usr_conf, logger): return
    
    env, agent = envs[0], agents[0]
    logger.info("Start Optimized Path Training...")
    start_t = time.time()

    base_map_data = read_map_data("conf/map_data/F_level_1.json")
    
    F_multi_stage = {}
    # 【与agent.py保持一致的优化顺序】
    target_seq = [1230, 620, 604, 2071, 3497, 3192, 2657, 2733, 2298, 1527, 759]

    for stage in range(11):
        target_pos = target_seq[stage]
        for pos_id_str, actions in base_map_data.items():
            current_pos = int(pos_id_str)
            new_state_id = current_pos * 11 + stage
            F_multi_stage[str(new_state_id)] = {}

            for act_str, transition in actions.items():
                next_pos, _, done = transition
                
                reward = -1 # 路径长度惩罚
                next_stage = stage
                if next_pos == target_pos:
                    reward = 500 # 提高奖励权重，确保DP收敛于此目标
                    if stage < 10:
                        next_stage = stage + 1
                
                next_combined_state = next_pos * 11 + next_stage
                F_multi_stage[str(new_state_id)][act_str] = [next_combined_state, reward, done]

    agent.learn(F_multi_stage)
    logger.info(f"Training completed in {time.time() - start_t} s")
    agent.save_model()
    return