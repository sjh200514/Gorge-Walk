#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import numpy as np
from kaiwu_agent.utils.common_func import create_cls, attached

# =================超参数=================
MAX_STEPS = 2000                   
URGENT_THRESHOLD_FACTOR = 3.5      
URGENT_GOAL_BOOST = 5.0            
REWARD_GOAL = 1000.0                
REWARD_TREASURE = 100.0             
PENALTY_STEP = -2.0                 
PENALTY_COLLISION = -10.0           
PENALTY_REPEAT = -1.5               

REWARD_FACTOR_TO_GOAL = 1.0         
REWARD_FACTOR_TO_TREASURE = 0.5     
REWARD_SPECIFIC_TREASURE_BOOST = 5.0 # 专注目标的额外奖励倍率

TREASURE_EFFECTIVE_DISTANCE = 30.0  
ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT = 0, 1, 2, 3

SampleData = create_cls("SampleData", state=None, action=None, reward=None, next_state=None)
ObsData = create_cls("ObsData", 
    feature=None, 
    raw_obs=None, 
    target_id=None, # 当前锁定的目标点索引(0-10)
    pos=None, 
    local_obstacle=None, 
    treasure_status=None,
    memory=None
)

@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]

def reward_shaping(frame_no, terminated, truncated, obs_data, _obs_data, score, prev_score, steps_in_episode, act):
    if terminated:
        return REWARD_GOAL, True

    reward = 0.0
    reward += PENALTY_STEP # 基础时间惩罚

    # 1. 拾取宝箱
    if score > prev_score:
        reward += REWARD_TREASURE

    # 2. 撞墙惩罚
    next_cell_idx = -1
    if act == ACTION_UP: next_cell_idx = 7
    elif act == ACTION_DOWN: next_cell_idx = 17
    elif act == ACTION_LEFT: next_cell_idx = 11
    elif act == ACTION_RIGHT: next_cell_idx = 13
    if next_cell_idx != -1 and obs_data.local_obstacle[next_cell_idx] == 1:
        reward += PENALTY_COLLISION

    # 3. 探索惩罚
    reward += _obs_data.memory[12] * PENALTY_REPEAT

    # 4. 目标引力逻辑 (与 state 逻辑强绑定)
    target_id = obs_data.target_id
    dist_prev = obs_data.raw_obs
    dist_now = _obs_data.raw_obs
    
    remaining_steps = MAX_STEPS - steps_in_episode
    # 判定紧急状态：没时间了或者没宝箱了
    is_urgent = (dist_now[0] * URGENT_THRESHOLD_FACTOR) > remaining_steps or target_id == 0

    if is_urgent:
        # 全力去终点
        reward += (dist_prev[0] - dist_now[0]) * REWARD_FACTOR_TO_GOAL * URGENT_GOAL_BOOST
    else:
        # 常规引导：趋向当前 target_id
        # 如果 target_id > 0，说明在找宝箱位
        d_p = dist_prev[target_id]
        d_n = dist_now[target_id]
        if d_n < TREASURE_EFFECTIVE_DISTANCE:
            reward += (d_p - d_n) * REWARD_FACTOR_TO_TREASURE * REWARD_SPECIFIC_TREASURE_BOOST
        
        # 始终保持对终点的基础引力
        reward += (dist_prev[0] - dist_now[0]) * REWARD_FACTOR_TO_GOAL

    return reward, is_urgent