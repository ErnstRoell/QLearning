import gym
import numpy as np
import matplotlib.pyplot as plt


######################################
### Set up env and constants
######################################

# Make gym and init q-table
env = gym.make("MountainCar-v0")
DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# General settings
EPISODES = 5000
SHOW_EVERY = 100

# DQLearn parameters  
LEARNING_RATE = 0.1
DISCOUNT = 0.95
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//10
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# For tracking the metrics of performance
episode_rewards = []
metrics = {'epsilon':[],'avg':[],'min':[],'max':[],'ep':[]} 


######################################
### Util functions
######################################


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table


######################################
### Q - Learning Functions
######################################
def update_q(q_table,discrete_state,action,result):
    new_state, reward, done, _ = result
    new_discrete_state = get_discrete_state(new_state)
    max_future_q = np.max(q_table[new_discrete_state])
    current_q = q_table[discrete_state+(action,)]
    new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE*(reward+DISCOUNT * max_future_q)
    q_table[discrete_state + (action,)]=new_q
    return q_table
#         env.render()
        

######################################
### Game loop
######################################

def run_game(q_table,epsilon):
    discrete_state = get_discrete_state(env.reset())
    done = False
    episode_reward = 0
    while not done:
        if np.random.random() < epsilon:
            action = np.random.randint(0,env.action_space.n)
        else:
            action = np.argmax(q_table[discrete_state])

        (*result,) = env.step(action)
        new_state, reward, done, _ = result
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        
        if not done:
            q_table =update_q(q_table,
                    discrete_state,
                    action,
                    result)

        elif new_state[0] >= env.goal_position:
            print(f"Goal reached at episode {episode}")
            q_table[discrete_state+(action,)]=0

        discrete_state = new_discrete_state
    return q_table, episode_reward

######################################
### Training
######################################

for episode in range(EPISODES):
    
    # Run the game once
    q_table, episode_reward = run_game(q_table, epsilon)
    
    # Update the epsilon value 
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    
    # Keep track of the metrics
    episode_rewards.append(episode_reward)
    metrics['epsilon'].append(epsilon)

    # Calculate a summary for the window we look at
    # Print also to console
    if not episode % SHOW_EVERY:
        avg_reward = sum(episode_rewards[-SHOW_EVERY:])/len(episode_rewards[-SHOW_EVERY:])
        metrics['ep'].append(episode)
        metrics['avg'].append(avg_reward)
        metrics['min'].append(min(episode_rewards[-SHOW_EVERY:]))
        metrics['max'].append(max(episode_rewards[-SHOW_EVERY:]))
        print(f'Episode: {episode} avg: {avg_reward}')

env.close()


######################################
### Visualization
######################################

plt.plot(metrics['ep'],metrics['avg'],label='avg')
plt.plot(metrics['ep'],metrics['min'],label='min')
plt.plot(metrics['ep'],metrics['max'],label='max')
plt.legend(loc=4)
plt.show()

