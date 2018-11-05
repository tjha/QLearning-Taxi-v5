# Tejas Jha 
# 5 November 2018
# EECS 498 - Reinforcement Learning HW 3
#
# --------------------------------------------------------------------------------------
# The code below implements a modified version of "Tabular Dyna-Q" as well as 
# implementation modifications to adapt to changes in the environemnt faster.
# This file also contains the code used to implement n-step semi-gradient TD estimation
#
# The functions below expand on the starter code provided that help generate the 
# states, actions, and rewards using Taxi-v4 and Taxi-v5 on openAi gym
# --------------------------------------------------------------------------------------

import numpy as np
import gym
import copy
import mytaxi
import math
import matplotlib.pyplot as plt

# Environment for Taxi-v4
ENV4 = gym.make('Taxi-v4').unwrapped
# Einvironment for Taxi-v5
ENV5 = gym.make('Taxi-v5').unwrapped
# Possible actions that can be taken
ACTIONS = [0,1,2,3,4,5]

# Helpers to choose best action given probability distributions
def _greedy(Q,s):
    qmax = np.max(Q[s])
    actions = []
    for i,q in enumerate(Q[s]):
        if q == qmax:
            actions.append(i)
    return actions

def greedy(Q,s):
    return np.random.choice(_greedy(Q,s))

def ep_greedy(Q,s,ep):
    if np.random.rand() < ep:
        return np.random.choice(len(Q[s]))
    else:
        return greedy(Q,s)

# Part (a) - Tabular Dyna-Q:
# Function implementation of dynaq to handle the stochastic nature of the environment
# returns Q and cum_steps - list of the cumulatative number of steps counted from the 
# first episode to the end of each episode.
def dynaq(env,n=10,gamma=1,alpha=1,epsilon=0.1,episodes=100):
    # Update kept on cumulative number of steps
    cum_steps = []
    Q = np.zeros((env.nS,env.nA))
    # Using deterministic model
    # Model = {}
    # for s in range(env.nS):
    #     Model[s] = ()
    # print(Model)
    #Model = np.zeros((env.nS,env.nA))
    # Loop over episodes
    #for i_episode in range(episodes):

    return Q, cum_steps

# Part (a) - adaptation of qlearn from hw2 to compare with Tabular Dyna-Q
# Key difference is 100 episodes instead of 500 by default
# Also, steps are now everaged through multiple callings of function
def qlearn(env,gamma=1,alpha=0.9,ep=0.05,runs=1,episodes=100):
    np.random.seed(3)
    env.seed(5)
    nS = env.nS
    nA = env.nA
    rew_alloc = []
    for run in range(runs):
        Q = np.zeros((nS,nA))
        rew_list = np.zeros(episodes)
        cum_steps = np.zeros(episodes)
        for idx in range(episodes):
            s = env.reset()
            done = False
            counter = 0
            cum_rew = 0
            while not done:
                a = ep_greedy(Q,s,ep)
                ss, r, done, _ = env.step(a)
                Q[s,a] = Q[s,a] + alpha * (r + gamma * np.max(Q[ss]) - Q[s,a])
                s = ss
                cum_rew +=  gamma**counter * r
                counter += 1.
            rew_list[idx] = cum_rew
            if idx == 0:
                cum_steps[idx] = counter
            else:
                cum_steps[idx] = counter + cum_steps[idx - 1]
        rew_alloc.append(rew_list)
    rew_list = np.mean(np.array(rew_alloc),axis=0)
    return Q, cum_steps


if __name__ == '__main__':

    # Part (a)
    print("Training using Tabular Dyna-Q for 100 episodes using Taxi-v4")
    # Average results over 20 runs
    #dynaq_Q_avg = 0
    #qlearn_Q_avg = 0
    dynaq_cum_steps_avg = np.zeros(shape=(100,))
    qlearn_cum_steps_avg = np.zeros(shape=(100,))

    for i in range(20):
        print("Performing run: " + str(i + 1))
        # Randomize seeds so runs are independent
        np.random.seed(i)
        ENV4.seed(i)
        _, dynaq_cum_steps = dynaq(ENV4)
        _, qlearn_cum_steps = qlearn(ENV4)
        # Update averages
        #dynaq_Q_avg += dynaq_Q / 20.0
        #qlearn_Q_avg += qlearn_Q / 20.0
        #dynaq_cum_steps_avg += np.divide(dynaq_cum_steps, 20.0)
        qlearn_cum_steps_avg += np.divide(qlearn_cum_steps, 20.0)
        break

    # Compare results with Q learning implementation used in hw2 in plot
    print(qlearn_cum_steps_avg)
    
