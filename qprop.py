import tensorflow as tf
import numpy as np
import gym
import tflearn

from replay_buffer import ReplayBuffer


class ActorNetwork(object):
	'''
	Deterministic policy mu: S --> A
	'''
	def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, a_output_activation='linear'):
		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_bound = action_bound
		self.learning_rate = learning_rate
		self.tau = tau
		self.a_output_activation = a_output_activation

		# Actor network
		self._input, self._out, self._scaled_out = self.create_actor_network()
		self._network_params = tf.trainable_variables()

		# Actor clone nework
		self._input_clone, self._out_clone, self._scaled_out_clone = self.create_actor_network()
		self._network_clone_params = tf.trainable_variables()[len(self._network_params):]

		

		# this gradient to be provided by critic nets
		self._action_gradient = tf.placeholder(tf.float32, [None, self.action_dim])

		self._actor_gradient = tf.gradients(self._scaled_out, self._network_params, -self._action_gradient)

		# optimizer
		self._optimizer = tf.train.AdamOptimizer(self.learning_rate).\
			apply_gradients(zip(self._actor_gradient, self._network_params))

		# Clone network update
		self._update_network_clone_params = \
			[self._network_clone_params[i].assign(
				tf.mul(self._network_params[i], tau) + tf.mul(self._network_clone_params[i], (1 - tau)))
				for i in range(len(self._network_clone_params))
			]

		self.num_trainable_vars = len(self._network_params) + len(self._network_clone_params)




	def create_actor_network(self):
		_inputs = tflearn.input_data(shape=[None, self.state_dim])
		_inputs_bn = tflearn.batch_normalization(_inputs)

		_net = tflearn.fully_connected(_inputs_bn, 400)
		_net = tflearn.batch_normalization(_net)
		_net = tflearn.activation(_net, 'relu')

		_net = tflearn.fully_connected(_net, 300)
		_net = tflearn.batch_normalization(_net)
		_net = tflearn.activation(_net, 'relu')

		_w_init = tflearn.initializations.uniform(minval=-3e-3, maxval=3e-3)
		# _out = tflearn.fully_connected(_net, self.action_dim, activation='tanh', weights_init=_w_init)
		_out = tflearn.fully_connected(_net, self.action_dim, 
				activation=self.a_output_activation, 
				weights_init=_w_init) 
		_scaled_out = tf.mul(_out, self.action_bound)

		return _inputs, _out, _scaled_out

	def train(self, state, a_gradient):
		self.sess.run(self._optimizer, feed_dict={
			self._input: state,
			self._action_gradient: a_gradient
		})

	def predict(self, inputs):
		return self.sess.run(self._scaled_out, feed_dict={
			self._input: inputs
		})

	def predict_clone(self, inputs):
		return self.sess.run(self._scaled_out_clone, feed_dict={
			self._input_clone: inputs
		})

	def update_network_clone(self):
		self.sess.run(self._update_network_clone_params)

	def get_num_trainable_vars(self):
		return self.num_trainable_vars


class CriticNetwork(object):
	'''
	Q:  S x A --> R
	'''
	def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars):
		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.learning_rate = learning_rate
		self.tau = tau


		# Critic Network
		self._input, self._action, self._out = self.create_critic_network()
		self._network_params = tf.trainable_variables()[num_actor_vars:]

		self._input_clone, self._action_clone, self._out_clone = self.create_critic_network()
		self._network_clone_params = tf.trainable_variables()[num_actor_vars + (len(self._network_params)):]

		# Clone network update
		self._update_network_clone_params = \
			[self._network_clone_params[i].assign(
				tf.mul(self._network_params[i], tau) + tf.mul(self._network_clone_params[i], (1 - tau)))
				for i in range(len(self._network_clone_params))
			]

		# network target (y_t)
		self._predicted_q_value = tf.placeholder(tf.float32, [None, 1])

		# define critic loss
		self._loss = tflearn.mean_square(self._predicted_q_value, self._out)
		self._optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self._loss)

		# Get the gradient w.r.t. the action
		self._action_grads = tf.gradients(self._out, self._action)

	def create_critic_network(self, l2w=1e-2):
		_input = tflearn.input_data(shape=[None, self.state_dim])
		_input_bn = tflearn.batch_normalization(_input)

		_action = tflearn.input_data(shape=[None, self.action_dim])
		_action_bn = tflearn.batch_normalization(_action)

		_net = tflearn.fully_connected(_input_bn, 400, weight_decay=l2w)
		_net = tflearn.batch_normalization(_net)
		_net = tflearn.activation(_net, 'relu')

		# Add the action tensor in the 2nd hidden layer
		# Use two temp layers to get the corresponding weights and biases
		t1 = tflearn.fully_connected(_net, 300, weight_decay=l2w)
		# t2 = tflearn.fully_connected(_action, 300, weight_decay=l2w)
		t2 = tflearn.fully_connected(_action_bn, 300, weight_decay=l2w)



		_net = tflearn.activation(tf.matmul(_net, t1.W) + tf.matmul(_action, t2.W) + t2.b)
		_net = tflearn.batch_normalization(_net)
		_net = tflearn.activation(_net, 'relu')

		# linear output layer
		_w_init = tflearn.initializations.uniform(minval=-3e-3, maxval=3e-3)
		_out = tflearn.fully_connected(_net, 1, weights_init=_w_init, weight_decay=1e-2)
		return _input, _action, _out

	def train(self, state, action, predicted_q_value):
		return self.sess.run([self._out, self._optimizer], feed_dict={
			self._input: state,
			self._action: action, 
			self._predicted_q_value: predicted_q_value
			})



	def predict(self, state, action):
		return self.sess.run(self._out, feed_dict={
			self._input: state,
			self._action: action
			})

	def predict_clone(self, state, action):
		return self.sess.run(self._out_clone, feed_dict={
			self._input_clone: state,
			self._action_clone: action
			})

	def action_gradients(self, state, action):
		return self.sess.run(self._action_grads, feed_dict={
			self._input: state,
			self._action: action
			})

	def update_network_clone(self):
		self.sess.run(self._update_network_clone_params)
