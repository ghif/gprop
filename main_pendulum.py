import tensorflow as tf
import gym
import numpy as np
from ddpg import *
import time

config = tf.ConfigProto(
	device_count = {'GPU': 0}
)

# ===========================
#   Utility Parameters
# ===========================
# Directory for storing gym results
MONITOR_DIR = 'results/gym_ddpg_cpu'
# Directory for storing tensorboard summary results
SUMMARY_DIR = 'results/tf_ddpg_cpu'

RESPATH = 'results/pendulum_ddpg_cpu'

# Render gym env during training

# Use Gym Monitor
GYM_MONITOR_EN = True
# Gym environment
ENV_NAME = 'Pendulum-v0'
RENDER_ENV = False
RANDOM_SEED = 1234

# ==========================
#   Training Parameters
# ==========================
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001

# Soft target update param
TAU = 0.001


# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64
RANDOM_SEED = 1234
# Discount factor 
GAMMA = 0.99
# Max training steps
MAX_EPISODES = 50000
# Max episode length
MAX_EP_STEPS = 1000


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
def train_ddpg(sess, env, actor, critic):
	# env: OpenAI environment
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
	elapsed_times = []

	for i in xrange(MAX_EPISODES):

		s = env.reset()

		ep_sum_reward = 0
		ep_avg_reward = 0
		ep_ave_max_q = 0

		
		start_t = time.time()
		for j in xrange(MAX_EP_STEPS):

			if RENDER_ENV: 
				env.render()

			# Added exploration noise
			a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i + j))

			s2, r, terminal, info = env.step(a[0])
			# print('[%d]. terminal : %d' % (j, terminal))
			

			replay_buffer.add(np.reshape(s, (actor.state_dim,)), np.reshape(a, (actor.action_dim,)), r, \
				terminal, np.reshape(s2, (actor.state_dim,)))

			# Keep adding experience to the memory until
			# there are at least minibatch size samples
			if replay_buffer.size() > MINIBATCH_SIZE:     
				s_batch, a_batch, r_batch, t_batch, s2_batch = \
					replay_buffer.sample_batch(MINIBATCH_SIZE)

				# Calculate targets
				a2_batch = actor.predict_clone(s2_batch)
				target_q = critic.predict_clone(s2_batch, a2_batch)

				y_i = []
				for k in xrange(MINIBATCH_SIZE):
					if t_batch[k]:
						y_i.append(r_batch[k])
					else:
						y_i.append(r_batch[k] + GAMMA * target_q[k])

				# Update the critic given the targets
				predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))
			
				ep_ave_max_q += np.amax(predicted_q_value)

				# Update the actor policy using the sampled gradient
				a_outs = actor.predict(s_batch)                
				grads = critic.action_gradients(s_batch, a_outs)
				actor.train(s_batch, grads[0])

				# Update target networks
				actor.update_network_clone()
				critic.update_network_clone()

			s = s2
			ep_sum_reward += r
			ep_avg_reward = ep_sum_reward / (j+1)



			if terminal:
				
				summary_str = sess.run(summary_ops, feed_dict={
					summary_vars[0]: ep_sum_reward,
					summary_vars[1]: ep_ave_max_q / float(j)
				})

				writer.add_summary(summary_str, i)
				writer.flush()
				elapsed_t = time.time() - start_t
				elapsed_times.append(elapsed_t)

				print '| Step-to-target: %d' % (j+1), '| Reward: %.2i' % int(ep_sum_reward), " | Episode", i, \
					'| Qmax: %.4f' % (ep_ave_max_q / float(j)), ' Elapsed time : %.3fs'% elapsed_t

				break
		# end for j MAX_EP

		steps.append((j+1))
		sum_rewards.append(ep_sum_reward)
		avg_rewards.append(ep_avg_reward)

		np.save(RESPATH, (steps, sum_rewards, avg_rewards, elapsed_times))

with tf.Session(config=config) as sess:
	env = gym.make(ENV_NAME)
	np.random.seed(RANDOM_SEED)
	tf.set_random_seed(RANDOM_SEED)
	env.seed(RANDOM_SEED)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	action_bound = env.action_space.high

	assert (env.action_space.high == -env.action_space.low)

	actor = ActorNetwork(sess, state_dim, action_dim, action_bound, 
		ACTOR_LEARNING_RATE, TAU, a_output_activation='tanh')

	critic = CriticNetwork(sess, state_dim, action_dim, 
		CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

	if GYM_MONITOR_EN:
		if not RENDER_ENV:
			env.monitor.start(MONITOR_DIR, video_callable=False, force=True)
		else:
			env.monitor.start(MONITOR_DIR, force=True)

	train_ddpg(sess, env, actor, critic)

	if GYM_MONITOR_EN:
		env.monitor.close()

tf.app.run()
