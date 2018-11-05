import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

import numpy as np
from numpy import array
import sys
from collections import defaultdict, OrderedDict

import gym
import mytaxi

env = gym.make('Taxi-v3').unwrapped
numS = env.observation_space.n
numA = env.action_space.n
print("#state:{}, #action{}".format(numS, numA))


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def mc_control_epsilon_greedy(env, num_episodes, epsilon=0.1, runs=10):
    np.random.seed(3)
    env.seed(5)

    runs = 10
    rew_alloc = []
    for run in range(runs):
        np.random.seed(run)
        env.seed(run)
        returns_sum = defaultdict(float)
        returns_count = defaultdict(float)

        
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
        policy = make_epsilon_greedy_policy(Q, epsilon, numA)
        rew_list = np.zeros(int(num_episodes/50))
        for i_episode in range(num_episodes):
            # Print out which episode we're on, useful for debugging.
            if i_episode % 1000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
                sys.stdout.flush()

            # Generate an episode.
            # An episode is an array of (state, action, reward) tuples
            episode = []
            state = env.reset()
            done = False
            for t in range(1000):
                probs = policy(state)
                action = np.random.choice(np.arange(len(probs)), p=probs)
                next_state, reward, done, _ = env.step(action)
                episode.append((state, action, reward))
                if i_episode % 50 == 0:
                    idx = int(i_episode / 50) 
                    rew_list[idx] += reward
                if done:
                    break
                state = next_state

            # Find all (state, action) pairs we've visited in this episode
            sa_in_episode = set([(x[0], x[1]) for x in episode])
            for state, action in sa_in_episode:
                sa_pair = (state, action)
                # Find the first occurance of the (state, action) pair in the episode
                first_occurence_idx = next(i for i,x in enumerate(episode)
                                           if x[0] == state and x[1] == action)
                # Sum up all rewards since the first occurrence
                G = sum([x[2] for x in episode[first_occurence_idx:]])
                # Update Q
                returns_sum[sa_pair] += G
                returns_count[sa_pair] += 1.0
                Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
        rew_alloc.append(rew_list)
    rew_list = np.mean(np.array(rew_alloc),axis=0)
    fig = plt.figure()
    plt.plot(rew_list)
    plt.savefig('mc_control_interim.eps')
    plt.close(fig)
        
    return Q, policy


if __name__ == '__main__':

    Q, policy = mc_control_epsilon_greedy(env, num_episodes=10000, epsilon=0.1, runs=10)
    

