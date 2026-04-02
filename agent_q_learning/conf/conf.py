#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


# Configuration of dimensions
# 关于维度的配置
class Config:

    #STATE_SIZE = 64 * 64 * 1024
    STATE_SIZE = 64 * 64 * 11
    ACTION_SIZE = 4
    LEARNING_RATE = 0.8
    GAMMA = 0.9
    EPSILON = 0.1
    
    EPISODES = 10000
    EPSILON_START = 1.0       # 探索率初始值
    EPSILON_END = 0.01        # 探索率最终值
    EPSILON_DECAY = 0.9995    # 每一步的衰减率 (这个值可以微调, 0.9995 是一个常用的起始值)


    # dimensionality of the sample
    # 样本维度
    SAMPLE_DIM = 5

    # Dimension of observation
    # 观察维度
    OBSERVATION_SHAPE = 250
