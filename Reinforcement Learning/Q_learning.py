import gymnasium as gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES =  20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0

if __name__ == "__main__":
    env = gym.envs.make("FrozenLake-v1")
    env.reset(seed=1)

    # You will need to update the Q_table in your iteration
    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()[0]

        ##########################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING ABOVE THIS LINE
        # Replace the following with Q-Learning

        while not done:
        
            rand_num = random.uniform(0, 1)
            if rand_num < EPSILON:
                action = env.action_space.sample()
            else:
                action_values = []
                for a in range(env.action_space.n):
                    value = Q_table[(obs, a)]
                    action_values.append((value, a))
                action = max(action_values)[1]

            results = env.step(action)
            new_obs = results[0]
            reward = results[1]
            terminated = results[2]
            truncated = results[3]

            done = terminated or truncated

            if not done:
                future_rewards = []
                for a in range(env.action_space.n):
                    future_reward = Q_table[(new_obs, a)]
                    future_rewards.append(future_reward)

                next_max = max(future_rewards)
                old_value = Q_table[(obs, action)]
                new_value = (1 - LEARNING_RATE) * old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max)
                Q_table[(obs, action)] = new_value
            else:
                old_value = Q_table[(obs, action)]
                new_value = (1 - LEARNING_RATE) * old_value + LEARNING_RATE * reward
                Q_table[(obs, action)] = new_value

            obs = new_obs
            episode_reward += reward

        EPSILON *= EPSILON_DECAY

        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE
        ##########################################################

        # record the reward for this episode
        episode_reward_record.append(episode_reward) 
     
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    #### DO NOT MODIFY ######
    model_file = open('Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
    #########################
