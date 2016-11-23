import numpy as np
import plotter

# RESPATH = 'results/oct_ddpg_v5.npy'
# RESPATH = 'results/ecs/oct_ddpg_v5.npy'
RESPATH = 'results/ecs/oct_ddpg_v5.npy'
(steps, sum_rewards, avg_rewards, sum_q, sum_max_q, elapsed_times) = np.load(RESPATH)

step_list_dict = {'DDPG' : steps}

lcolor_avg_dict = {'DDPG': [1,0,0]}
lcolor_dict = {'DDPG': [1,0.7,0.7]}

plotter.plot_oct_single(
	step_list_dict, step_list_dict,
	title='Octopus Arm',
	ylabel='steps to target',
	imgfile='ddpg_steps_v5.png',
	axisrange=[0, 3e5, 0, 300],
	lcolor_dict = lcolor_dict,
	lcolor_avg_dict = lcolor_avg_dict
)
