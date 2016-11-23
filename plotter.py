import matplotlib
from numpy import *
import matplotlib.pyplot as plt

def plot_loss(losses,filename='QProp_ContinuousBandit.png'):
	l = len(losses)
	x = range(0,l)
	y = losses
	plt.plot(x,y,linewidth=4.0)
	plt.legend(['QProp'], loc='upper right')

	plt.title('Continuos Bandit')
	plt.xlabel('Steps')
	plt.ylabel('Loss')
	plt.grid() #grid on

	# set font
	font = {'family' : 'normal',
    	'weight' : 'bold',
    	'size'   : 16}

	matplotlib.rc('font',**font)

	plt.savefig(filename)
	plt.show()


def plot_oct_agent(oct_agent, filename='oct_agent_result'):
	x = range(0,len(oct_agent.steps_per_ep))
	y = oct_agent.steps_per_ep
	
	plt.plot(x,y,'ro')
	plt.legend(['QProp'], loc='upper right')

	plt.title('Octopus Arm')
	plt.xlabel('Episodes')
	plt.ylabel('Steps to Target')
	
	plt.grid() #grid on

	# set font
	font = {'family' : 'normal',
    	'weight' : 'bold',
    	'size'   : 16}

	matplotlib.rc('font',**font)

	plt.savefig(filename)
	plt.show()

def plot_oct_outputs_icml(oct_outputs, imgfile='oct_plot.png'):

	## To access the information
	# - steps_to_target : oct_outputs['steps_to_target']
	# - average rewards : oct_outputs['rew_avg_per_ep']
	# - total rewards : oct_outputs['rew_tot_per_ep']

	avg_num = 16
	nsteps = oct_outputs["steps_to_target"]
	n = len(nsteps)
	display_length = (n / avg_num) * avg_num
	csteps = reshape(nsteps[:display_length], (display_length/avg_num, avg_num))
	y = mean(csteps, axis = 1)
	x = cumsum(sum(csteps, axis = 1))

	plt.plot(x,y,'-')
	plt.legend(['QProp'], loc='upper right')

	plt.title('Octopus Arm')
	plt.xlabel('Steps')
	plt.ylabel('Steps to Target')
	
	plt.grid() #grid on
	plt.axis([0, 3e5, 0, 300])

	# set font
	font = {'family' : 'normal',
    	'weight' : 'bold',
    	'size'   : 16}

	matplotlib.rc('font',**font)

	plt.savefig(imgfile)
	plt.show()

def plot_oct_outputs_multi(
	rlist, vlist,
	title='[title]',
	ylabel='steps to target',
	imgfile='oct_plot.png',
	axisrange=[0, 3e5, 0, 300],
	plotter = None, show=True,
	good_id = None,
	lcolor = 'gray',
	lcolor_avg = 'red'
	):
	if plotter is None:
		plotter = plt

	# store the average in every multiple of nf
	nf = 5000
	MAX_X = 300000

	# plot all lists
	ylist = []
	x = range(0,MAX_X+1,nf)

	for j in range(0,len(rlist)):
		l1 = 0
		prev_mod = 0
		cur_mod = 0
		rlist_avg = []
		rlist_avg.append(vlist[j][0])
		for i in range(1, len(rlist[j])+1):
			sum_r = sum(rlist[j][:i])
			cur_mod = mod(sum_r,nf)
			if cur_mod < prev_mod: 
				
				l2 = i-1
				# print '[multi]',i,':',sum_r,' -> ',len(rlist[j][l1:l2])
				rlist_avg.append(mean(vlist[j][l1:l2]))
				l1 = l2


			prev_mod = cur_mod

		
		
		y = rlist_avg
		print 'Plotting Line : ',j, '(length: ',len(y),')'
		
		plotter.plot(x,y,color=lcolor)

		ylist.append(y)

		
	# end for
	# plot the average
	y = mean(ylist,axis=0)
	line_avg = plotter.plot(x, y , color=lcolor_avg,linewidth=2)



	# plt.legend(['QProp'], loc='upper right')

	plotter.title(title)
	plotter.xlabel('# training actions')
	plotter.ylabel(ylabel)
	
	plotter.grid() #grid on
	plotter.axis(axisrange)
	# plt.axis([0, len(x), 0, 300])

	# set font
	font = {'family' : 'normal',
    	'weight' : 'bold',
    	'size'   : 16}

	matplotlib.rc('font',**font)

	if show:
		plotter.savefig(imgfile)
		plotter.show()

	return plotter

def plot_oct_single(
	xdict, ydict,
	title='[title',
	ylabel='steps to target',
	imgfile='oct_plot.png',
	axisrange=[0, 3e5, 0, 300],
	lcolor_dict=None,
	lcolor_avg_dict=None
	):

	# store the average in every multiple of nf
	nf = 5000
	MAX_X = int(axisrange[1])

	x = range(0, MAX_X+1, nf)

	# print(x)

	for key, xlist in xdict.iteritems():
		ylist = ydict[key]
		print(xlist)
		print(ylist)
		print('len : %d, %d' % (len(xlist), len(ylist)))

		lcolor = lcolor_dict[key]
		lcolor_avg = lcolor_avg_dict[key]

		l1 = 0
		prev_mod = 0
		cur_mod = 0
				
		y_avg_list = [300]
		for i in range(1, len(xlist)+1):
			
			sum_r = sum(xlist[:i])

			cur_mod = mod(sum_r,nf)
			if cur_mod < prev_mod: 
				
				l2 = i-1
				# print(i, ':', sum_r, '->', mean(ylist[l1:l2]))
				y_avg_list.append(mean(ylist[l1:l2]))

				# print '[multi]',i,':',sum_r,' -> ',len(rlist[j][l1:l2])
				# rlist_avg.append(mean(vlist[j][l1:l2]))
				l1 = l2

				if sum_r >= MAX_X:
					
					break;

			prev_mod = cur_mod
		
	
	label = lcolor_dict.keys()[0]
	color = lcolor_avg_dict[label]
	plt.plot(x, y_avg_list, label=label, color=color, linewidth=2)

	plt.legend()
	plt.xlabel('# training actions')
	plt.ylabel(ylabel)
	
	plt.grid() #grid on
	plt.axis(axisrange)
	# plt.axis([0, len(x), 0, 300])

	# set font
	font = {'family' : 'normal',
    	'weight' : 'bold',
    	'size'   : 16}



	matplotlib.rc('font',**font)

	
	plt.savefig(imgfile)
	plt.show()
		

def plot_oct_outputs_multi2(
	rdict, vdict,
	title='[title]',
	ylabel='steps to target',
	imgfile='oct_plot.png',
	axisrange=[0, 3e5, 0, 300],
	lcolor_dict = None,
	lcolor_avg_dict = None
	):
	

	# store the average in every multiple of nf
	nf = 5000
	MAX_X = 300000

	# plot all lists
	
	x = range(0,MAX_X+1,nf)

	line_avgs = []
	for key, rlist in rdict.iteritems():
		ylist = []
		vlist = vdict[key]
		lcolor = lcolor_dict[key]
		lcolor_avg = lcolor_avg_dict[key]

		for j in range(0,len(rlist)):
			l1 = 0
			prev_mod = 0
			cur_mod = 0
			rlist_avg = []
			rlist_avg.append(vlist[j][0])
			print(len(rlist[j]))
			print(len(vlist[j]))
			print(vlist[j][0])
			for i in range(1, len(rlist[j])+1):
				sum_r = sum(rlist[j][:i])
				cur_mod = mod(sum_r,nf)
				if cur_mod < prev_mod: 
					l2 = i-1
					# print '[multi]',i,':',sum_r,' -> ',len(rlist[j][l1:l2])
					rlist_avg.append(mean(vlist[j][l1:l2]))
					l1 = l2

				prev_mod = cur_mod			
			
			y = rlist_avg
			print 'Plotting Line : ',j, '(length: ',len(y),')'

			print(x)
			print(y)
			print(len(x))
			print(len(y))
			print(lcolor)
			
			plt.plot(x,y,color=lcolor)

			ylist.append(y)

			
		# end for
		# plot the average
		y = mean(ylist,axis=0)
		if key == 'qprop':
			legend_lbl = 'GradProp'
		elif key == 'copdac':
			legend_lbl = 'COPDAC-Q'
		else:
			legend_lbl = key

		line_avg, = plt.plot(x, y , label=legend_lbl, color=lcolor_avg,linewidth=2)
		line_avgs.append(line_avg)
		
	plt.legend(handles=line_avgs)
	
	


	# plt.legend(['QProp'], loc='upper right')

	plt.title(title)
	plt.xlabel('# training actions')
	plt.ylabel(ylabel)
	
	plt.grid() #grid on
	plt.axis(axisrange)
	# plt.axis([0, len(x), 0, 300])

	# set font
	font = {'family' : 'normal',
    	'weight' : 'bold',
    	'size'   : 16}

	matplotlib.rc('font',**font)

	
	plt.savefig(imgfile)
	plt.show()



	
def plot_oct_outputs(oct_outputs, imgfile='oct_plot.png'):

	## To access the information
	# - steps_to_target : oct_outputs['steps_to_target']
	# - average rewards : oct_outputs['rew_avg_per_ep']
	# - total rewards : oct_outputs['rew_tot_per_ep']


	y = oct_outputs['steps_to_target']
	x = range(0,len(y))
	

	plt.plot(x,y,'ro')
	plt.legend(['QProp'], loc='upper right')

	plt.title('Octopus Arm')
	plt.xlabel('Episodes')
	plt.ylabel('Steps to Target')
	
	plt.grid() #grid on

	# set font
	font = {'family' : 'normal',
    	'weight' : 'bold',
    	'size'   : 16}

	matplotlib.rc('font',**font)

	plt.savefig(imgfile)
	plt.show()