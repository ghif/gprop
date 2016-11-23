import tensorflow as tf
# import gym
import env

import numpy as np
from ddpg import *

import sys

import time

config = tf.ConfigProto(
	device_count = {'GPU': 0}
)


# ===========================
#   Utility Parameters
# ==========================='
# Directory for storing tensorboard summary results
SUMMARY_DIR = 'results/tf_oct_ddpg_v7'

RESPATH = 'results/oct_ddpg_v7'

# Render gym env during training

# ==========================
#   Training Parameters
# ==========================
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 1e-4
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 1e-3

# Soft target update param
TAU = 1e-3

# Size of replay buffer
BUFFER_SIZE = 1e6
MINIBATCH_SIZE = 64
RANDOM_SEED = 1234
# Discount factor 
GAMMA = 0.99

# Max training steps
MAX_EPISODES = 50000
# Max episode length
MAX_EP_STEPS = 300
# MAX_TOT_STEPS = 30000
MAX_STEP_COUNTER = 300000


SIGMA = 3. # variance of the exploration
REW_GAIN = 30
# REW_GAIN = 50

def min_dist_to_target(state, x_target, y_target):
	'''
	- state: the vector of state variables (the information of the end-of-arm coordinate sits here !)
	- (x_target, y_target): the target coordinate 
	'''
	dim = len(state)
	dim_h = dim/2;

	x_edge = range(6)
	y_edge = range(6)
	for i in range(6):
		#print "%d %d" % (2 + i*4, 26 + i*4)
		#print "%d %d" % (3 + i*4, 27 + i*4)
		x_edge[i] = (state[2 + i*4] + state[26 + i*4]) / 2
		y_edge[i] = (state[3 + i*4] + state[27 + i*4]) / 2

	x_comp = range(2)
	y_comp = range(2)
	dist = range(2)
	for i in range(2):
		#print "%d %d" %(i+4, i+3)
		x_comp[i] = (x_edge[i+4] + x_edge[i+3]) / 2
		y_comp[i] = (y_edge[i+4] + y_edge[i+3]) / 2
		dist[i] = np.sqrt((x_comp[i] - x_target)**2 + (y_comp[i] - y_target)**2)

	return min(dist)

# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries(): 
	episode_reward = tf.Variable(0.)
	reward_summary = tf.scalar_summary("Reward", episode_reward)


	episode_ave_max_q = tf.Variable(0.)
	qmax_summary = tf.scalar_summary("Qmax Value", episode_ave_max_q)

	summary_vars = [episode_reward, episode_ave_max_q]
	# summary_ops = tf.merge_all_summaries()
	summary_ops = tf.merge_summary([reward_summary, qmax_summary])


	return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================
def train_ddpg(sess, envr, actor, critic):
	# env: Java environment (Octopus Arm)
	# actor: ActorNetwork
	# critic: CriticNetwork

	# Set up summary Ops
	summary_ops, summary_vars = build_summaries()

	sess.run(tf.initialize_all_variables())
	writer = tf.train.SummaryWriter(SUMMARY_DIR, sess.graph)

	# Initialize target network weights
	actor.update_network_clone()
	critic.update_network_clone()

	# Initialize replay memory
	replay_buffer = ReplayBuffer(BUFFER_SIZE)

	sum_rewards = []
	avg_rewards = []
	steps = []
	sum_q = []
	sum_max_q = []
	elapsed_times = []

	for i in range(MAX_EPISODES):
		env.send_str('START')

		data = env.receive(2+envr.dim_state)
		terminal = int(data[0])
		start_flag = True

		# get state (s_t) and compute action (a_t)
		# print(data)
		s = map(float, data[2:])
		
		s = np.array(s)
		# print('state : ', s)

		# closest_distance = min_dist_to_target(s, envr.target_x, envr.target_y)
		# print('-- init closest distance : ', closest_distance)

		ep_sum_reward = 0.
		ep_avg_reward = 0.

		ep_sum_max_q = 0,
		ep_avg_max_q = 0.
		ep_sum_avg_q = 0.
		
		step_counter = 0


		start_t = time.time()

		while not terminal:
			step_counter += 1

			# compute action with exploration noise
			mu = actor.predict(np.reshape(s, (1, s.shape[0])))
			mu = np.reshape(mu, (mu.shape[1], ))
			epsilon = np.random.randn(envr.dim_action)
			factor = SIGMA / step_counter
			a = mu + factor * epsilon

			# # print('det. action (',mu.shape,') : ', mu)
			# print('state_t', s)
			# print('det action_t : ', mu)
			# print('action_t : ', a)
		
			# compute next state
			env.send_str('STEP')
			env.send_str(str(envr.dim_action))

			env.send_action(a)
			data = env.receive(3 + envr.dim_state)

			s2 = map(float, data[3:])
			s2 = np.array(s2)

			terminal = int(data[1])
			
			# compute reward
			closest_distance = min_dist_to_target(s, envr.target_x, envr.target_y)
			new_closest_distance = min_dist_to_target(s2, envr.target_x, envr.target_y)

			if start_flag:
				closest_distance = new_closest_distance
				start_flag = False

			r = REW_GAIN * (closest_distance - new_closest_distance) # non-hit reward

			# if (new_closest_distance + 1e-2) < closest_distance:
			# 	r = 1
			# else:
			# 	r = -1

			if terminal: #hitting the target
				if step_counter < MAX_EP_STEPS:
					r = float(data[2])
					print('-- hit the target (r = %f) ' % r)
			
			# print('(cd, ncd, r) : (%.2f, %.2f, %f)' % (closest_distance, new_closest_distance, r))

			replay_buffer.add(np.reshape(s, (actor.state_dim,)), np.reshape(a, (actor.action_dim,)), r, \
				terminal, np.reshape(s2, (actor.state_dim,)))

			if replay_buffer.size() > MINIBATCH_SIZE:
				s_batch, a_batch, r_batch, t_batch, s2_batch = \
					replay_buffer.sample_batch(MINIBATCH_SIZE)

				# calculate targets
				a2_batch = actor.predict_clone(s2_batch)
				target_q = critic.predict_clone(s2_batch, a2_batch)

				y_i = []
				for k in range(MINIBATCH_SIZE):
					if t_batch[k]:
						y_i.append(r_batch[k])
					else:
						y_i.append(r_batch[k] + GAMMA * target_q[k])

				# Update the critic given the targets
				predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))
				# print(predicted_q_value)

				ep_sum_avg_q += np.average(predicted_q_value)
				ep_sum_max_q += np.amax(predicted_q_value)
				ep_avg_max_q = ep_sum_max_q / step_counter

				# Update the actor policy using the sampled gradient
				a_outs = actor.predict(s_batch)
				grads = critic.action_gradients(s_batch, a_outs)
				actor.train(s_batch, grads[0])

				# update target networks
				actor.update_network_clone()
				critic.update_network_clone()


			s = s2
			
			ep_sum_reward += r
			ep_avg_reward = ep_sum_reward / step_counter

			if terminal:
				# print('step to target : %f' % step_counter)
				# print('ep_avg_max_q : %f' % ep_avg_max_q)

				# summary_str = sess.run(summary_ops, feed_dict={
				# 	summary_vars[0]: ep_sum_reward,
				# 	summary_vars[1]: ep_avg_max_q
				# })

				# writer.add_summary(summary_str, i)
				# writer.flush()
				elapsed_t = time.time() - start_t
				elapsed_times.append(elapsed_t)

				steps.append(step_counter)
				print '| Step-to-target: %d' % step_counter, ' Total step : %d' % np.sum(steps), ' | Reward: %.2i' % int(ep_sum_reward), " | Episode", i, \
					'| Q: %.4f' % ep_sum_avg_q,  '| Qmax: %.4f' % ep_avg_max_q, '| Elapsed time: %.1fs' % elapsed_t

				break
		# end while not terminal

		sum_rewards.append(ep_sum_reward)
		avg_rewards.append(ep_avg_reward)

		sum_q.append(ep_sum_avg_q)
		sum_max_q.append(ep_avg_max_q)


		np.save(RESPATH, (steps, sum_rewards, avg_rewards, sum_q, sum_max_q, elapsed_times))

		if np.sum(steps) > MAX_STEP_COUNTER:
			break;
	# end for i MAX_EPISODES

with tf.Session(config=config) as sess:
	# env = gym.make(ENV_NAME)
	envr = env.OctopusArm()
	print(envr)


	# np.random.seed(RANDOM_SEED)
	# tf.set_random_seed(RANDOM_SEED)
	# env.seed(RANDOM_SEED)

	# state_dim = env.observation_space.shape[0]
	# action_dim = env.action_space.shape[0]
	# action_bound = env.action_space.high

	state_dim = envr.dim_state
	action_dim = envr.dim_action
	action_bound = 1.

	actor = ActorNetwork(sess, state_dim, action_dim, action_bound, 
		ACTOR_LEARNING_RATE, TAU, a_output_activation='sigmoid')

	critic = CriticNetwork(sess, state_dim, action_dim, 
		CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

	train_ddpg(sess, envr, actor, critic)


tf.app.run()