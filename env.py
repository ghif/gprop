
'''
env.py
- This code provides the interface between python agent and Java environments
'''
import socket
import sys
import os
from threading import Thread

LINE_SEPARATOR = '\n'
BUF_SIZE = 4096 #in bytes
PORT = 32000 #default: 8888
#PORT = 8888
# MODE = 'external_gui' # with visualization
MODE = 'external' # without visualization

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)



#envpath = '/am/phoenix/home1/gif/Desktop/PhD/Writing/GitRepo/fishertca/rl-framework/environments' 
envpath = 'rl-framework/environments' 

xmlconf = 'settings1.xml'
#xmlstate = 'initialStates_gif.xml'
xmlstate = 'initialStates.xml'

def connect(host='localhost', port=PORT):
	try:
		sock.connect((host, port))
	except socket.error:
		print 'Unable to contact environment at the given host/port.'
		sys.exit(1)
        
def send_str(s):
    sock.send(s + LINE_SEPARATOR)
    
def receive(numTokens):
    data = ['']
    # print '[receive] numTokens : ',numTokens
    while len(data) <= numTokens:
        # print '[receive] data : ',data
        srev = sock.recv(BUF_SIZE)
        # print 'srev : ',srev
        rawData = data[-1] + srev
        del data[-1]
        data = data + rawData.split(LINE_SEPARATOR)
        
    del data[-1]
    return data
    
def send_action(action):
    #sends all the components of the action one by one
    for a in action:
        send_str(str(a))

def init_octopus_env():
    javapath = 'java -Djava.endorsed.dirs='+envpath+'/octopus-arm/lib -jar '+envpath+'/octopus-arm/octopus-environment.jar '+MODE+' '+envpath+'/octopus-arm/'+xmlconf+' '+envpath+'/octopus-arm/'+xmlstate+' '+str(PORT)
    os.system(javapath)

def init_octopus_rotating_env():
    javapath = 'java -Djava.endorsed.dirs='+envpath+'/octopus-arm-rotating/lib -jar '+envpath+'/octopus-arm-rotating/octopus-environment.jar '+MODE+' '+envpath+'/octopus-arm-rotating/'+xmlconf+' '+envpath+'/octopus-arm-rotating/'+xmlstate+' '+str(PORT)
    os.system(javapath)

def init_mountaincar_env():
	# please change the below filepath if the jar file is in another place
    os.system('java -jar ../rl-framework/environments/mountain-car/mountaincar-environment.jar random 8888') #start with a random car position
    #os.system('java -jar ./environments/mountain-car/mountaincar-environment.jar fixed -0.5 8888') #start with a specific car position

def init_puddleworld_env():
    os.system('java -jar ./environments/puddle-world/puddleworld-environment.jar random 8888') #start with a random position
    #os.system('java -jar ./environments/puddle-world/puddleworld-environment.jar fixed 0.5 0.2 8888') #start with a specific position


class MountainCar:
	def __init__(self):
		# connect to the Java environment
		thread = Thread(target = init_mountaincar_env,args = [])
		print 'Please wait while initializing the environment'
		thread.start()
		thread.join(2)

		connect() #localhost: 8888 (default)

		send_str('GET_TASK')
		data = receive(2)
		print 'Data : ',data
		self.dim_state = int(data[0])
		self.dim_action = 1
		print 'The state space is %d dimensional and the action space is %d dimensional' % (self.dim_state, self.dim_action)

class OctopusArm:
    def __init__(self):
        # connect to the Java environment
        thread = Thread(target = init_octopus_rotating_env,args = [])
        print 'Please wait while initializing the environment'
        thread.start()
        thread.join(2)

        connect(host='localhost',port=PORT) #localhost: 8888 (default)

        send_str('GET_TASK')
        data = receive(2)
        print 'Data : ',data
        self.dim_state = int(data[0])
        self.dim_action = int(data[1])

        # coordinate of the oct. arm's target, must match with one specified in the xml conf file
        self.target_x = -3.25
        self.target_y = -3.25
