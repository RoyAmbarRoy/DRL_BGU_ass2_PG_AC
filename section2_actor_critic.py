import gym
import numpy as np
import tensorflow as tf
import collections

env = gym.make('CartPole-v1')
env._max_episode_steps = None

np.random.seed(1)


class StateValueNetwork:
    def __init__(self, state_size, output_size, learning_rate, name='state_value_network'):
        self.state_size = state_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            # self.total_reward = tf.placeholder(tf.int32, name="total_reward")
            self.td_target = tf.placeholder(tf.float32, name="td_target")
            # self.total_reward = tf.placeholder(tf.float32, name="total_reward")

            self.W1 = tf.get_variable("W1", [self.state_size, 12],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, 1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.value_estimate = tf.squeeze(self.Z2)

            # # Softmax probability distribution over actions
            # self.reward_expectation = tf.squeeze(tf.nn.li(self.output))
            # Loss with negative log probability
            # self.td_error = self.r + GAMMA * self.v_ - self.v

            self.loss = tf.losses.mean_squared_error(self.value_estimate, self.td_target)  # the loss function

            # self.mse_loss = tf.losses.mean_squared_error(self.total_reward,
            #                                              self.value_estimate)  # the loss function
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.A = tf.placeholder(tf.float32, name="advantage")

            self.W1 = tf.get_variable("W1", [self.state_size, 12],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, self.action_size],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=self.output,
                                                                        labels=self.action)  # (y_hat, y)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.A)
            self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


# Define hyper parameters
state_size = 4
action_size = env.action_space.n

max_episodes = 5000
max_steps = 5000
discount_factor = 1
learning_rate = 0.001
value_net_learning_rate = 0.01

render = False

# Initialize the policy network
tf.reset_default_graph()
policy = PolicyNetwork(state_size, action_size, learning_rate)

state_value_network = StateValueNetwork(state_size, 1, value_net_learning_rate)

# Start training the agent with REINFORCE algorithm
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    solved = False
    episode_rewards = np.zeros(max_episodes)
    average_rewards = 0.0

    for episode in range(max_episodes):
        state = env.reset()
        state = state.reshape([1, state_size])
        episode_transitions = []

        i = 1
        # X_policy = []
        # Y_policy = []
        #
        # X_value_net = []
        # Y_value_net = []

        for step in range(max_steps):
            # choose action from policy network given initial state
            actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)

            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape([1, state_size])

            # update statistics
            episode_rewards[episode] += reward

            # calc advantage
            # sess = sess or tf.get_default_session()
            # V_s = sess.run(state_value_network.value_estimate, {state_value_network.state: state})
            # V_s_prime = sess.run(state_value_network.value_estimate, {state_value_network.state: next_state})


            # else:  # not done
            #     V_s_prime = sess.run(state_value_network.value_estimate, {state_value_network.state: next_state})

            # calc advantage
            sess = sess or tf.get_default_session()
            V_s = sess.run(state_value_network.value_estimate, {state_value_network.state: state})
            V_s_prime = sess.run(state_value_network.value_estimate, {state_value_network.state: next_state})

            td_target = reward + discount_factor * V_s_prime
            td_error = td_target - V_s  # the TD error

            # update state V value network
            state_value_feed_dict = {state_value_network.state: state,
                                     state_value_network.td_target: td_target}
            sess = sess or tf.get_default_session()
            _, state_value_loss = sess.run([state_value_network.optimizer, state_value_network.loss],
                                           state_value_feed_dict)
            # print(state_value_loss)

            # update policy network
            action_one_hot = np.zeros(action_size)
            action_one_hot[action] = 1
            feed_dict = {policy.state: state,
                         policy.A: td_error,
                         policy.action: action_one_hot}
            sess = sess or tf.get_default_session()
            _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)

            if done:  # episode done
                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],
                                                                                   round(average_rewards, 2)))
                if average_rewards > 475:
                    print(' Solved at episode: ' + str(episode))
                    solved = True
                break

            # re-assign
            i = discount_factor * i
            state = next_state

        if solved:
            break
