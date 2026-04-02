#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_agent.utils.common_func import Frame
from kaiwu_agent.utils.common_func import attached
import time
import math
from agent_q_learning.feature.definition import (
    sample_process,
    reward_shaping,
)
import os
from tools.train_env_conf_validate import check_usr_conf, read_usr_conf
from tools.metrics_utils import get_training_metrics
from agent_q_learning.conf.conf import Config


@attached
def workflow(envs, agents, logger=None, monitor=None):
    env, agent = envs[0], agents[0]
    EPISODES = Config.EPISODES

    # Initializing monitoring data
    # 监控数据初始化
    monitor_data = {
        "reward": 0,
        "diy_1": 0,
        "diy_2": 0,
        "diy_3": 0,
        "diy_4": 0,
        "diy_5": 0,
    }

    last_report_monitor_time = time.time()

    logger.info("Start Training ...")
    start_t = time.time()

    total_rew, win_cnt = (
        0,
        0,
    )

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = read_usr_conf("agent_q_learning/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_q_learning/conf/train_env_conf.toml")
        return

    # check_usr_conf is a tool to check whether the game configuration is correct
    # It is recommended to perform a check before calling reset.env
    # check_usr_conf会检查游戏配置是否正确，建议调用reset.env前先检查一下
    valid = check_usr_conf(usr_conf, logger)
    if not valid:
        logger.error("check_usr_conf return False, please check")
        return

    episode_rewards = []
    episode_steps = []
    agent.epsilon = Config.EPSILON_START

    for episode in range(EPISODES):
        # Retrieving training metrics
        # 获取训练中的指标
        training_metrics = get_training_metrics()
        if training_metrics:
            logger.info(f"training_metrics is {training_metrics}")

        # Reset the environment and obtain the initial state
        # 重置环境, 并获取初始状态
        obs, state = env.reset(usr_conf=usr_conf)

        agent.current_target_id = None

        # Disaster recovery
        # 容灾
        if obs is None:
            continue

        # First frame processing
        # 首帧处理
        obs_data = agent.observation_process(obs, state)

        # Task loop
        # 任务循环
        done = False
        # 初始化 prev_score 用于计算分数增量
        prev_score = 0
        steps_in_episode = 0
        while not done:
            # Agent performs inference to obtain the predicted action for the next frame
            # Agent 进行推理, 获取下一帧的预测动作
            # agent.epsilon = max(0.1, agent.epsilon * math.exp(-(1 / EPISODES) * episode))
            act_data, model_version = agent.predict(list_obs_data=[obs_data])
            act_data = act_data[0]

            # Unpacking ActData into actions
            # ActData 解包成动作
            act = agent.action_process(act_data)

            # Interact with the environment, perform actions, and obtain the next state
            # 与环境交互, 执行动作, 获取下一步的状态
            frame_no, _obs, score, terminated, truncated, state = env.step(act)
            if _obs is None:
                break

            # Feature processing
            # 特征处理
            _obs_data = agent.observation_process(_obs, state)

            # Compute reward
            # 计算 reward
            reward, is_urgent = reward_shaping(frame_no, terminated, truncated, obs_data, _obs_data, score, prev_score, steps_in_episode, act)

            # Determine over and update the win count
            # 判断结束, 并更新胜利次数
            done = terminated or truncated
            if terminated:
                win_cnt += 1

            # Updating data and generating frames for training
            # 数据更新, 生成训练需要的 frame
            sample = Frame(
                state=obs_data.feature,
                action=act,
                reward=reward,
                next_state=_obs_data.feature,
            )

            # Sample processing
            # 样本处理
            sample = sample_process([sample])

            # train
            # 训练
            agent.learn(sample)

            # Update total reward and state
            # 更新总奖励和状态
            total_rew += reward
            obs_data = _obs_data
            prev_score = score
            steps_in_episode += 1

            # 在每一步的最后，对 agent.epsilon 进行衰减
            if agent.epsilon > Config.EPSILON_END:
                agent.epsilon *= Config.EPSILON_DECAY

        episode_steps.append(steps_in_episode)
        episode_rewards.append(total_rew)

        # Reporting training progress
        # 上报训练进度
        now = time.time()
        if now - last_report_monitor_time > 60:
            slice_size = min(len(episode_rewards), 50)
            avg_reward = sum(episode_rewards[-slice_size:]) / slice_size if slice_size > 0 else 0
            avg_steps = sum(episode_steps[-slice_size:]) / slice_size if slice_size > 0 else 0
            win_rate = win_cnt / (episode + 1) if episode + 1 > 0 else 0
            logger.info(f"Episode: {episode + 1}, Epsilon: {agent.epsilon:.4f}, Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.1f}, Win Rate: {win_rate:.4f}, Is Urgent: {is_urgent}")
            #logger.info(f"Episode: {episode + 1}, Reward: {total_rew}")
            #logger.info(f"Training Win Rate: {win_cnt / (episode + 1)}")
            monitor_data["reward"] = total_rew
            monitor_data["diy_1"] = agent.epsilon
            monitor_data["diy_2"] = win_rate
            monitor_data["diy_3"] = avg_reward
            monitor_data["diy_4"] = avg_steps
            monitor_data["diy_5"] = 1 if is_urgent else 0
            if monitor:
                monitor.put_data({os.getpid(): monitor_data})

            episode_rewards.clear()
            episode_steps.clear()
            
            total_rew = 0
            last_report_monitor_time = now

        # The model has converged, training is complete, and reporting monitoring metric
        # 模型收敛, 结束训练, 上报监控指标
        if win_cnt / (episode + 1) > 0.9 and episode > 100:
            logger.info(f"Training Converged at Episode: {episode + 1}")
            monitor_data["reward"] = total_rew
            monitor_data["diy_1"] = agent.epsilon
            monitor_data["diy_2"] = win_rate
            monitor_data["diy_3"] = avg_reward
            monitor_data["diy_4"] = avg_steps
            monitor_data["diy_5"] = 1 if is_urgent else 0
            if monitor:
                monitor.put_data({os.getpid(): monitor_data})
            break

    end_t = time.time()
    logger.info(f"Training Time for {episode + 1} episodes: {end_t - start_t} s")
    agent.episodes = episode + 1

    # model saving
    # 保存模型
    agent.save_model()

    return
