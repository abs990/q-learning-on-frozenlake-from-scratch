"""
Q-Learning on FrozenLake from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_q_table
import numpy as np

def init_q_table(num_states, num_actions):
    """Return a zero-initialized Q-table of shape (num_states, num_actions)."""
    # TODO: build a 2D float64 numpy array of zeros sized by states and actions.
    return np.zeros((num_states, num_actions), dtype=np.float64)

# Step 2 - max_q_value
import numpy as np

def max_q_value(q_table, state):
    """Return the maximum Q value across all actions for the given state."""
    return np.max(q_table[state, :])

# Step 3 - greedy_action
import numpy as np

def greedy_action(q_table, state):
    """Return the action index with the highest Q value at the given state."""
    return int(np.argmax(q_table[state]))

# Step 4 - sample_random_action
def sample_random_action(action_space):
    return int(action_space.sample())

# Step 5 - should_explore
def should_explore(epsilon, rng):
    """Return True with probability epsilon using the provided numpy Generator."""
    return rng.random() < epsilon

# Step 6 - epsilon_greedy_action
import numpy as np

def epsilon_greedy_action(q_table, state, epsilon, action_space, rng):
    """Return an epsilon-greedy action for the given state."""
    if should_explore(epsilon, rng):
        return sample_random_action(action_space)
    else:
        best_actions = np.flatnonzero(q_table[state] == q_table[state].max())
        return int(rng.choice(best_actions))

# Step 7 - decay_epsilon
def decay_epsilon(epsilon, decay_rate, min_epsilon):
    return max(min_epsilon, decay_rate * epsilon)

# Step 8 - td_target
def td_target(reward, gamma, q_table, next_state, done):
    if not done:
        reward += gamma * max_q_value(q_table, next_state)
    return reward

# Step 9 - td_error
def td_error(target, q_table, state, action):
    return (target - q_table[state][action])

# Step 10 - q_learning_update
def q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma):
    # apply Q(s,a) += alpha * (target - Q(s,a)) in place and return the new Q value
    error = td_error(td_target(reward, gamma, q_table, next_state, done), q_table, state, action)
    q_table[state, action] += alpha * error
    return q_table[state, action]

# Step 11 - interaction_step
def interaction_step(env, q_table, state, epsilon, alpha, gamma, rng):
    # select epsilon-greedy action, step env, apply Q-learning update, return (next_state, reward, done)
    # pick action
    best_action = epsilon_greedy_action(q_table, state, epsilon, env.action_space, rng)
    # update env
    next_state ,reward ,terminated ,truncated ,info = env.step(best_action)
    reward = float(reward)
    done = terminated or truncated
    # q learning update
    q_learning_update(q_table, state, best_action, reward, next_state, done, alpha, gamma)
    return next_state, reward, done

# Step 12 - run_training_episode (not yet solved)
# TODO: implement

# Step 13 - train_q_learning (not yet solved)
# TODO: implement

# Step 14 - extract_greedy_policy (not yet solved)
# TODO: implement

# Step 15 - run_greedy_episode (not yet solved)
# TODO: implement

# Step 16 - evaluate_success_rate (not yet solved)
# TODO: implement

