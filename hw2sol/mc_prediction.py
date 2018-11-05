import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import sys

from collections import defaultdict, OrderedDict
matplotlib.style.use('ggplot')

# import RidiculusTaxi
import mytaxi

env = gym.make('Taxi-v3').unwrapped
numS = env.observation_space.n
numA = env.action_space.n
print("#state:{}, #action{}".format(numS, numA))

def rms(a,b):
    return np.sqrt(np.mean((a-b)**2))


def action_s(policy_s):
    p = policy_s / sum(policy_s)
    return np.random.choice(numA,p=p)


def mc_prediction(baseline, policy, env, num_episodes):

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final value function
    V = np.zeros(env.nS)
    RMS = []
    for i_episode in range(1, num_episodes + 1):

        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        done=False
        while not done:  
            action = action_s(policy[state])
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        # Find all states the we've visited in this episode
        states_in_episode = set([x[0] for x in episode])
        for state in states_in_episode:
            # Find the first occurance of the state in the episode
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state)
            # Sum up all rewards since the first occurance
            G = sum([x[2] for i,x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[state] += G
            returns_count[state] += 1.0
            V[state] = returns_sum[state] / returns_count[state]
        RMS_ep = rms(baseline,V)
        RMS.append(RMS_ep)

    return np.array(RMS), V

def run_mc(env,baseline,runs,policy, num_episodes=1000):
    np.random.seed(3)
    env.seed(5)
    
    RMS = np.zeros(num_episodes)
    V = np.zeros(env.nS)
    
    RMS,V = mc_prediction(baseline, policy, env, num_episodes=num_episodes)
    
    fig = plt.figure()
    plt.plot(RMS)
    plt.savefig('mc-error.eps')
    plt.close(fig)
    fig = plt.figure()
    plt.plot(V,marker='o',linestyle='None',label='mc')
    plt.plot(baseline,marker='x',linestyle='None',label='base')
    plt.legend(loc=3)
    plt.savefig('mc-qplot.eps')
    plt.close(fig)


if __name__ == '__main__':

    policy = np.load('policy.npy')
    baseline = np.load('baseline.npy')

    
    run_mc(env,baseline,runs=1,policy=policy,num_episodes=50000)

    