import numpy as np
from vizdoom import *
import random
import matplotlib.pyplot as plt
from scipy import stats
import cv2
from scipy.ndimage.interpolation import shift
import sys
import math
import scipy.io as sio
noisy = True

def create_environment():
    game = DoomGame()
    game.load_config('vizdoom/myConfig.cfg')  
    game.set_doom_scenario_path('vizdoom/square2.wad')
    #game.set_doom_scenario_path('vizdoom/tmaze.wad')
    #game.set_doom_map('map01')
    game.set_mode(Mode.PLAYER)
    game.set_window_visible(False)
    game.set_automap_buffer_enabled(True)
    game.set_automap_mode(WHOLE)
    game.set_available_buttons([MOVE_LEFT, MOVE_RIGHT, MOVE_FORWARD, MOVE_BACKWARD, TURN_RIGHT]) 
    #game.set_available_buttons([MOVE_FORWARD, MOVE_BACKWARD, TURN_RIGHT, TURN_LEFT]) 
    game.set_available_game_variables([POSITION_X, POSITION_Y, ANGLE, VELOCITY_X, VELOCITY_Y, CAMERA_FOV])
    game.init()     
    return game

def turn90(game):
    deg = round((game.get_game_variable(ANGLE) - 90)/90)*90
    if deg < 0:
        deg += 360
        game.make_action([0, 0, 0, 0, 1], 10)
    while game.get_game_variable(ANGLE) > deg+1:
        game.make_action([0, 0, 0, 0, 1], int(math.ceil((game.get_game_variable(ANGLE) - deg)/10)))
    #print(game.get_game_variable(ANGLE))

def lookAround(game):
    vision = game.get_state().screen_buffer
    turn90(game)
    vision = np.concatenate((vision, game.get_state().screen_buffer))
    turn90(game)
    vision = np.concatenate((vision, game.get_state().screen_buffer))
    turn90(game)
    vision = np.concatenate((vision, game.get_state().screen_buffer))
    turn90(game)

    return vision

### plotting ###
def compute_receptive_field(act, x, y, bin_size):
    # batch size and the number of neurons
    num_neuron = act.shape[0];
    # sampling size for gaussian smoothing
    sample_size = int(np.minimum(bin_size[0], bin_size[1]) / 4);
    if sample_size % 2 == 0 :
        sample_size += 1;
    
    # receptive field
    receptive_fields = np.zeros((num_neuron, bin_size[1], bin_size[0]));
    # compute receptive field with spatial position
    map_x = x.reshape(-1);
    map_y = y.reshape(-1);
    for neuron_idx in np.arange(num_neuron):
        f = act[neuron_idx, :].reshape(-1);
        ratemap, _, _, _ = stats.binned_statistic_2d(map_y, map_x, f, 
                                                     statistic = lambda x : np.mean(x),
                                                     bins = [bin_size[1], bin_size[0]]);
        # Nan value complete
        ratemap_nan = np.copy(ratemap);
        nan_idx = np.where(np.isnan(ratemap_nan));
        
        # gaussian sampling 
        ratemap_nan[nan_idx] = 0.0;
        ratemap_nan = cv2.GaussianBlur(ratemap_nan, (sample_size, sample_size), 0);
        ratemap[nan_idx] = ratemap_nan[nan_idx];
        receptive_fields[neuron_idx] = ratemap;
    return receptive_fields;

if noisy:
    fpath="exp22-noisy/"
else:
    fpath="exp2/"
def plot_receptive_field(activity, traj, plot_size=[1,1], bin_length=50, fig_size=[10,10], fname="untitled"):
    ratemaps = compute_receptive_field(activity, traj[:, 0], traj[:, 1], [int(traj[:, 0].size/bin_length), int(traj[:, 1].size/bin_length)]);
    simData = {'r': ratemaps}
    sio.savemat(fpath+fname+'.mat', simData, appendmat=False)
    num_neuron = ratemaps.shape[0];
    fig = plt.figure(figsize=fig_size);
    for neuron_idx in range(plot_size[0]*plot_size[1]):
        plt.subplot(plot_size[0], plot_size[1], neuron_idx + 1);
        plt.imshow(np.flipud(ratemaps[neuron_idx]), interpolation = 'gaussian');
    #plt.show();
    fig.savefig(fpath+fname+".png")

def plot_traj(traj, env_range=[(-200, 200), (-200, 200)], fname="traj"):
    fig = plt.figure();
    plt.subplot(1, 1, 1);
    plt.plot(traj[:, 0], traj[:, 1], 'ko-');
    plt.xlim(env_range[0]);
    plt.ylim(env_range[1]);
    plt.title('Trajectory');
    #plt.show();
    fig.savefig(fpath+fname+".png")


### primary visual cortex ###

def downsample(img, size):
    res = np.zeros([int(img.shape[0]/size), int(img.shape[1]/size)])
    for y in range(int(img.shape[0]/size)):
        for x in range(int(img.shape[1]/size)):
            res[y, x] = np.average(img[y*size: (y+1)*size, x*size: (x+1)*size])
    return res

class V1:
    def __init__(self, initState, kernel_size=20):   
        self.kernel_size = kernel_size
        self.height = initState.shape[0] # input image size
        self.width = initState.shape[1]
        self.vn_height = int(self.height/self.kernel_size*2)
        self.vn_width = int(self.width/self.kernel_size*2)
        self.nNeurons = self.vn_height * self.vn_width
        self.data = np.zeros([0, self.nNeurons])
        
    def response(self, state, angle=0.0): 
        #print(state) # shape: (480, 640)
        #cv2.imwrite('state_original.jpg', state);        
        # Gabor filter
        filter_outputs = np.zeros([0, self.vn_height, self.vn_width])
        for i in range(4):
            g_kernel = cv2.getGaborKernel((self.kernel_size, self.kernel_size), sigma=8.0, theta=np.pi/4.0*i, lambd=15.0, gamma=1.5, psi=0)
            #cv2.imwrite('kernel.jpg', g_kernel*255/np.max(g_kernel));
            filter_output = downsample(cv2.filter2D(state, cv2.CV_8UC3, g_kernel), int(self.kernel_size/2))
            filter_outputs = np.append(filter_outputs, filter_output.reshape([1, self.vn_height, self.vn_width]), axis=0)
            #cv2.imwrite('filter_output'+str(i)+'.jpg', filter_output);
        filtered_state = np.amax(filter_outputs, axis=0).reshape([self.vn_height, self.vn_width])
        #cv2.imwrite('state_filtered.jpg', filtered_state);
        
        # 2D to 1D
        filtered_state = filtered_state.reshape((1, -1)) # shape => (1, vn_height * vn_width)
        # sum normalization (-> length unity)
        filtered_state = filtered_state / np.max(filtered_state)
        filtered_state = filtered_state / np.tile(np.linalg.norm(filtered_state, axis=1).reshape((-1, 1)), (1, filtered_state.shape[1]))
        
        #self.data = np.append(self.data, filtered_state, axis=0)
        return filtered_state


### Grid Cell Sheet ###
sin60 = np.sqrt(3)/2.0

class GCSheet:
    def __init__(self, N, Ncpc=0, alpha=1.0, alphaC=0, eta=0.1):
        self.Nx = int(round(np.sqrt(N/sin60)))
        self.Ny = int(round(self.Nx * sin60))
        self.N = self.Nx * self.Ny
        #print(self.Nx, self.Ny)
        
        self.poses = [[i+1, j+1] for i in range(self.Nx) for j in range(self.Ny)] #poses[n-1] = [int(n/self.Ny)+1, n%self.Ny]
        self.alpha = alpha/400 #np.random.uniform(1.0/400, 2.6/400) # smaller -> larger spacing # [1,3]
        #print(self.alpha*400)
        self.beta = np.random.uniform(0, np.pi/3)
        self.R = np.array([[np.cos(self.beta), -np.sin(self.beta)], [np.sin(self.beta), np.cos(self.beta)]])
        
        self.A = np.random.uniform(0, 1/np.sqrt(self.N), self.N) # initial activity
        self.updateW([0, 0]) # initial weights
        
        self.alphaC = alphaC
        self.eta = eta
        self.Wcpc = np.random.rand(Ncpc, N)
        self.Wcpc /= np.tile(np.sqrt(np.sum(self.Wcpc * self.Wcpc, axis = 0)), (Ncpc, 1))
        
    def distTri(self, i, j, v): # i, j are [x,y], where 1<=x<=Nx, 1<=y<=Ny
        ci = np.array([(i[0]-0.5)/self.Nx, (i[1]-0.5)/self.Ny*sin60])
        cj = np.array([(j[0]-0.5)/self.Nx, (j[1]-0.5)/self.Ny*sin60])
        s = np.array([[0,0],[-0.5,sin60],[-0.5,-sin60],[0.5,sin60],[0.5,-sin60],[-1,0],[1,0]])
        return np.min(np.linalg.norm(np.tile(ci, (7,1)) - np.tile(cj, (7,1)) + np.tile(v, (7,1)) + s, axis=1))
        
    def dynamic(self, Acpc=None):
        tau = 0.8
        B = np.matmul(self.w, self.A.T)
        if Acpc is not None:
            self.A = (1-self.alphaC) * np.clip((1-tau)*B + tau*B/np.sum(self.A), 0, None) + self.alphaC * np.dot(Acpc, self.Wcpc).reshape((-1)) / self.Wcpc.shape[0] # normalize by Ncpc
        else:
            self.A = np.clip((1-tau)*B + tau*B/np.sum(self.A), 0, None)
        
    def updateW(self, v): # v: [vx, vy]
        I = 0.3
        T = 0.05
        sigma = 0.25 # unit: meter # 0.24
        
        v = self.alpha * np.matmul(self.R, np.array(v).T)
        self.w = np.array([[I * np.exp(-(self.distTri(i,j,v)**2)/sigma**2) - T for i in self.poses] for j in self.poses])
        self.w *= np.ones(self.w.shape) - np.identity(self.N) # eliminate self-recurrent?
        
    def trainCPC(self, Acpc):
        self.Wcpc += self.eta * np.repeat(Acpc.reshape([1,-1]).T, self.A.shape[0], axis=1) * np.repeat(self.A.reshape([1,-1]), Acpc.shape[1], axis=0)
        self.Wcpc /= np.sqrt(np.sum(self.Wcpc * self.Wcpc, axis = 0))

class GCpop:
    def __init__(self, N, Nsheets, Ncpc=0, alphaC=0, eta=0.1):
        self.Nneurons = N*Nsheets
        self.sheets = []
        for i in range(Nsheets):
            self.sheets.append(GCSheet(N, Ncpc, 1.0+1.6/(Nsheets-1)*i,alphaC, eta))
            for t in range(300):
                self.sheets[-1].dynamic()
        self.data = np.zeros((0, self.Nneurons))
                
    def updateW(self, v):
        for s in self.sheets:
            s.updateW(v)
            
    def dynamic(self, Acpc=None, record=False, training=False):
        for s in self.sheets:
            s.dynamic(Acpc)
            if Acpc is not None and training:
                s.trainCPC(Acpc)
        res = np.array([s.A for s in self.sheets]).reshape((1, self.Nneurons))
        if record:
            self.data = np.append(self.data, res, axis=0)
        return res


### visual/motion place cells (with competitive learning) ###

class PlaceCell: 
    def __init__(this, N, nDim, eta=0.2, inFilter=1, noise=0.003): 
        this.inFilter = 1.0-inFilter # for MPC
        this.noise = noise
        this.eta = eta;
        #this.mapSize = mapSize; # 2D size of number of place fields
        this.nDim = nDim; # number of pixels of visual input; number of motion grid cells
        this.nNodes = N
        
        # create weight space
        this.w = np.random.rand(this.nDim, this.nNodes);
        # normalize the weight
        this.w = this.w / np.tile(np.sqrt(np.sum(this.w * this.w, axis = 0)), (this.nDim, 1));
        
        this.data = np.zeros([0, this.nNodes])

    def train(this, sample, nIter=1):
        for iterIdx in range(nIter):
            winIdx, neuronOutput = this.computeWinNeuron(sample);
            if this.inFilter>0:
                this.updateWeight(np.clip(sample-np.amax(sample)*this.inFilter*np.ones(sample.shape), 0, None), winIdx, neuronOutput);
            else:
                this.updateWeight(sample, winIdx, neuronOutput);

    def computeWinNeuron(this, neuronInput):
        # forward output
        neuronOutput = np.dot(neuronInput, this.w)
        # add some noise
        winIdx = np.argmax(neuronOutput + np.amax(neuronOutput) * this.noise * this.nNodes * np.random.rand(1, this.nNodes));
        return winIdx, neuronOutput;
    
    def updateWeight(this, neuronInput, winIdx, neuronOutput):
        # on-line update the winning neuron weight
        this.w[:, winIdx: winIdx+1] += this.eta * neuronInput.T * neuronOutput[0, winIdx];
        # normalization
        if np.sum(this.w[:, winIdx]) != 0:
            this.w[:, winIdx] = this.w[:, winIdx] / np.sqrt(np.sum(this.w[:, winIdx] * this.w[:, winIdx]));
        
    def estimate(this, x, e=0.0, record=False):
        # forward output
        y = np.dot(x, this.w);
        # E% max inhibition
        y = y - e * np.tile(np.reshape(np.max(y, 1), [-1, 1]), [1, y.shape[1]]);
        y = y * (y > 0);
        y = y / (1.0 - e);
        if record:
            this.data = np.append(this.data, y, axis=0)
        return y;


### Conjunctive Place Cell ###

class CPC: 
    def __init__(self, N, nVPC, nMPC, eta=1, noise=0.003, alpha=0.3, inFilter=1):
        self.alpha = alpha
        self.obj = PlaceCell(N, nVPC+nMPC, eta=eta, noise=noise, inFilter=inFilter)

    def train(self, actV, actM, nIter=1):
        self.obj.train(np.concatenate((self.alpha*actV, (1-self.alpha)*actM), axis=1), nIter)
                 
    def estimate(self, actV, actM, e=0.0, record=False):
        return self.obj.estimate(np.concatenate((self.alpha*actV, (1-self.alpha)*actM), axis=1), e=e, record=record)


### goal cell ###
class GoalCell:
    def __init__(self, nPC):
        self.nPC = nPC # number of place cells
        self.w = np.zeros(nPC)
        self.gc_max = 0
        self.data = []
    
    def response(self, PCact, record=False):
        res = np.dot(self.w, PCact.reshape((-1)))
        if record:
            self.data.append(res)
        return res
        
    def memorizeReward(self, rat, tRange = 50, decay = 0.8):
        # synapses from currently firing place cells are enhanced
        ripple = rat.Acpc.reshape((-1))/np.linalg.norm(rat.Acpc, axis=1)
        #sys.stdout.write('ripple max: '+str(np.amax(ripple))+' ')
        self.w += ripple
        for t in range(tRange):
            ripple = rat.triggeredResponse(ripple).reshape(-1)
            ripple /= np.linalg.norm(ripple, axis=0)
            #sys.stdout.write(str(np.amax(ripple))+' ')
            self.w +=  ripple * (decay**(t+1))
        self.w /= np.linalg.norm(self.w, axis=0)
        self.gc_max = self.response(rat.Acpc)
        print("gc_max=", self.gc_max)
    
    def forget(self, factor=1000):
        self.w /= factor
        self.gc_max /= factor
        
#def sim_gc(gc, PCpop, env_range, n, plot=True, bin_length=50, fig_size=[5,5]):
#    def f(pos):
#        return gc.response([pc.response(pos, False) for pc in PCpop.MPCs], False)
#    return simulate_plot([f], env_range, n, plot, bin_length, fig_size)
#
#def simulate_plot(funcs, env_range, n, plot=True, bin_length=50, fig_size=[15,15]):
#    traj_x = np.random.rand(n) * (env_range[0][1]-env_range[0][0]) + np.ones((n)) * env_range[0][0]
#    traj_y = np.random.rand(n) * (env_range[1][1]-env_range[1][0]) + np.ones((n)) * env_range[1][0]
#    traj = np.array([traj_x, traj_y]).T
#    
#    activity = np.array([[func(pos) for pos in traj] for func in funcs])
#    if plot:
#        plot_receptive_field(activity, traj, plot_size=[int(len(funcs)**0.5), int(len(funcs)**0.5)], bin_length=bin_length, fig_size=fig_size)
#    return traj, activity


### Wrapper for the whole system ###

class System:
    def __init__(self, initState, nSteps=5, Ncpc=196):
        self.nSteps = nSteps
        ## V1 ##
        self.v1 = V1(initState)
        self.Av1 = self.v1.response(initState)
        
        ## Visual Place Cells ##
        self.vpc = PlaceCell(121, self.v1.nNeurons, eta=40, noise=0.008)
        self.Avpc = self.vpc.estimate(self.Av1, e=0.95)
        
        ## Motion Grid Cells ##
        self.mgc = GCpop(90, 5, Ncpc=Ncpc, alphaC=0.03, eta=50) # noisy: 0.05, normal: 0.10
        self.Amgc = self.mgc.dynamic()
        
        ## Motion Place Cells ##
        self.mpc = PlaceCell(225, self.mgc.Nneurons, eta=1, inFilter=0.8, noise=0.004)
        self.Ampc = self.mpc.estimate(self.Amgc, e=0.8)
        
        ## Conjuctive Place Cells ##
        self.cpc = CPC(Ncpc, self.vpc.nNodes, self.mpc.nNodes, eta=50, noise=0.005, alpha=0.3)
        self.Acpc = self.cpc.estimate(self.Avpc, self.Ampc, e=0.3)
        self.rw = np.zeros([Ncpc, Ncpc]) # recurrent weights
        self.cnt = 0

        ## Goal Cell ##
        self.gc = GoalCell(Ncpc)
        self.globPos = [0,0]

    def trainHopfield(self):
        n = self.Acpc.shape[1]
        rw = np.matmul(self.Acpc.T, self.Acpc)
        mask = np.ones((n, n)) - np.identity(n)
        #rw = rw*mask
        if np.linalg.norm(rw.reshape(-1), axis=0) != 0:
            self.rw = ( self.rw*self.cnt + rw/np.linalg.norm(rw.reshape(-1), axis=0) )/(self.cnt+1)
            self.cnt += 1
        
    def showRW(self):
        fig = plt.figure();
        plt.imshow(self.rw);
        #plt.show();
        fig.savefig(fpath+"rw.png")
    
    def triggeredResponse(self, Acpc):
        return np.matmul(Acpc.reshape((1, -1)), self.rw)

    def makeAction(self, game, v, training=False, trainHop=False, recording=False):
        if noisy:
            v = np.array(v)*13.62/self.nSteps
        else:
            v = (np.array([game.get_game_variable(POSITION_X), game.get_game_variable(POSITION_Y)]) - np.array(self.globPos)) / self.nSteps
            self.globPos = [game.get_game_variable(POSITION_X), game.get_game_variable(POSITION_Y)]
        ## conjuctive & recurrent ##
        if training:
            self.cpc.train(self.Avpc, self.Ampc, nIter=10)
        self.Acpc = self.cpc.estimate(self.Avpc, self.Ampc, e=0.3, record=recording)
        if trainHop:
            self.trainHopfield()
        
        ## visual pathway ##
        #self.Av1 = self.v1.response(game.get_state().screen_buffer)
        self.Av1 = self.v1.response(lookAround(game))
        if training:
            self.vpc.train(self.Av1, nIter=20)
        self.Avpc = self.vpc.estimate(self.Av1, e=0.95, record=recording)
        
        ## self-motion pathway ##
        self.mgc.updateW(v)
        for step in range(self.nSteps):
            self.Amgc = self.mgc.dynamic(self.Acpc, record=recording, training=training)
        if training:
            self.mpc.train(self.Amgc, nIter=5)
        self.Ampc = self.mpc.estimate(self.Amgc, e=0.8, record=recording)



### moving functions ###

def move(game, action, speed=1):
    game.make_action(action, speed)
    v = [game.get_game_variable(VELOCITY_X), game.get_game_variable(VELOCITY_Y)]
    game.make_action([0]*game.get_available_buttons_size(), 50)
    return v

def randMove(game, avoidWall=False, speed=1):
    actions = np.identity(game.get_available_buttons_size(), dtype = int)
    mask = np.ones(game.get_available_buttons_size(), dtype=bool)
    if mask.size > 4:
        mask[4] = False # TURN_RIGHT
    if avoidWall:
        pos = [game.get_game_variable(POSITION_X), game.get_game_variable(POSITION_Y)]
        if pos[0] > 150:
            mask[2] = False # inhibit MOVE_FORWARD
        if pos[0] < -150:
            mask[3] = False # inhibit MOVE_BACKWARD
        if pos[1] > 150:
            mask[0] = False # inhibit MOVE_LEFT
        if pos[1] < -150:
            mask[1] = False # inhibit MOVE_RIGHT
    actions = actions[mask].tolist()
    action = random.choice(actions)
    return move(game, action, speed), action

def go_back_turn(game, ori_dir, speed=1):
    actions = np.identity(game.get_available_buttons_size(), dtype = int)
    if actions.size > 4:
        actions = np.delete(actions, 4, axis=0) #TURN_RIGHT
    opposite = ori_dir
    
    if ori_dir[0]:
        opposite[0] = 0
        opposite[1] = 1
        actions = np.delete(actions, 0, axis=0)
    elif ori_dir[1]:
        opposite[0] = 1
        opposite[1] = 0
        actions = np.delete(actions, 1, axis=0)
    elif ori_dir[2]:
        opposite[2] = 0
        opposite[3] = 1
        actions = np.delete(actions, 2, axis=0)
    elif ori_dir[3]:
        opposite[2] = 1
        opposite[3] = 0
        actions = np.delete(actions, 3, axis=0)
    
    return move(game, opposite, speed), random.choice(actions.tolist())

def initPos(game, rat):
    # go to another initial position
    for i in range(100):
        v, direction = randMove(game, True, 6)
        rat.makeAction(game, v, training=False)
    
def search(game, rat, rewardPos, tolerance):
    rewarded = False
    pos = [game.get_game_variable(POSITION_X), game.get_game_variable(POSITION_Y)]
    while not rewarded:
        if pos[0]>rewardPos[0]-tolerance and pos[0]<rewardPos[0]+tolerance and pos[1]>rewardPos[1]-tolerance and pos[1]<rewardPos[1]+tolerance: 
            print("reward! (random search)", pos)
            rat.gc.memorizeReward(rat, tRange=10, decay=0.9)
            #rat.gc.memorizeReward(rat, tRange=5, decay=0.1)
            break
        v, direction = randMove(game, True, random.randint(3,6))
        pos = [game.get_game_variable(POSITION_X), game.get_game_variable(POSITION_Y)]
        rat.makeAction(game, v, training=False)


### explicit strategy ###
    
def exploitation(game, rat, rewardPos, tolerance, nMoves=100, trial=""):
    rewarded = False
    # random initial position
    pos = [game.get_game_variable(POSITION_X), game.get_game_variable(POSITION_Y)]
    traj = [pos]
    Agc = rat.gc.response(rat.Acpc, record=True) 
    
    speed = 1
    p = 0.01
    threshold_low = rat.gc.gc_max*0.7
    threshold_high = rat.gc.gc_max*0.95
    #print(threshold_low, threshold_high)
    v, direction = randMove(game, True) # randomly decide a direction
    pos = [game.get_game_variable(POSITION_X), game.get_game_variable(POSITION_Y)]
    rat.makeAction(game, v, training=False)
    traj.append(pos)
    
    for i in range(nMoves):
        ori_pos = pos
        Agc = rat.gc.response(rat.Acpc, record=True) 
        if Agc < threshold_high or len(rat.gc.data) < 3:
            if Agc < threshold_low:
                #sys.stdout.write("fast ")
                speed = 5
            else:
                #sys.stdout.write("medium ")
                speed = 3
            
            if rat.gc.data[-2] < Agc: # current position is better than the last; i.e. direction is good
                #sys.stdout.write("following gradient ")
                v = move(game, direction, speed)
                rat.makeAction(game, v, training=False)
            else:
                #sys.stdout.write("going back and turn ")
                v, direction = go_back_turn(game, direction, speed)
                rat.makeAction(game, v, training=False)
                v = move(game, direction, speed)
                rat.makeAction(game, v, training=False)
        
        else:
            #sys.stdout.write("slow ")
            speed = 1
        
            if rat.gc.data[-2] < Agc or rat.gc.data[-3] < Agc: # current position is better than the last; i.e. direction is good
                #sys.stdout.write("following gradient ")
                v = move(game, direction, speed)
                rat.makeAction(game, v, training=False)
            else:
                #sys.stdout.write("going back and turn ")
                v, direction = go_back_turn(game, direction, speed)
                rat.makeAction(game, v, training=False)
                v = move(game, direction, speed)
                rat.makeAction(game, v, training=False)
        
        pos = [game.get_game_variable(POSITION_X), game.get_game_variable(POSITION_Y)]
        traj.append(pos)
        #print(ori_pos, gc_activity)
        
        if pos[0]>rewardPos[0]-tolerance and pos[0]<rewardPos[0]+tolerance and pos[1]>rewardPos[1]-tolerance and pos[1]<rewardPos[1]+tolerance: 
            print("reward!", i)
            rewarded = True
            break
    if not rewarded:
        print("fail!")
    plot_traj(np.array(traj), fname="exploit"+trial)
    return rewarded

def ratGo(game, rat, rewardPos, tolerance, trial=""):
    if not exploitation(game, rat, rewardPos, tolerance, nMoves=100, trial=trial):
        # if failed to find the reward in nMoves, the rat believes the reward position has changed
        print("forget and search...")
        rat.gc.forget()
        search(game, rat, rewardPos, tolerance/1.5)

### unified strategy ###
def unifiedStrategy(game, rat, rewardPos, tolerance, trial=""):
    rewarded = False
    rat.gc.data = []
    # initial position & responses
    pos = [game.get_game_variable(POSITION_X), game.get_game_variable(POSITION_Y)]
    traj = [pos]
    Agc = rat.gc.response(rat.Acpc, record=True) 
    
    speed = 1
    p = np.exp(-rat.gc.gc_max) # possibility to random walk
    threshold_low = rat.gc.gc_max*0.7 #0.7
    threshold_high = rat.gc.gc_max*0.95 #0.95
    
    v, direction = randMove(game, True) # randomly decide a direction
    pos = [game.get_game_variable(POSITION_X), game.get_game_variable(POSITION_Y)]
    rat.makeAction(game, v, training=False)
    traj.append(pos)
    
    while not rewarded:
        ori_pos = pos
        Agc = rat.gc.response(rat.Acpc, record=True) 
        
        if random.random() < p:
            v, direction = randMove(game, avoidWall=True, speed=random.randint(3,6))
        else:
            if Agc < threshold_high or len(rat.gc.data) < 3:
                if Agc < threshold_low:
                    #sys.stdout.write("fast ")
                    speed = 5
                else:
                    #sys.stdout.write("medium ")
                    speed = 3
                
                ##### far away -- random walk or follow gradient? #####
                
                if rat.gc.data[-2] < Agc: # current position is better than the last; i.e. direction is good
                    #sys.stdout.write("following gradient ")
                    v = move(game, direction, speed)
                    rat.makeAction(game, v, training=False)
                else:
                    #sys.stdout.write("going back and turn ")
                    v, direction = go_back_turn(game, direction, speed)
                    rat.makeAction(game, v, training=False)
                    v = move(game, direction, speed)
                    rat.makeAction(game, v, training=False)
            
            else:
                #sys.stdout.write("slow ")
                speed = 1
            
                if rat.gc.data[-2] < Agc or rat.gc.data[-3] < Agc: # current position is better than the last; i.e. direction is good
                    #sys.stdout.write("following gradient ")
                    v = move(game, direction, speed)
                    rat.makeAction(game, v, training=False)
                else:
                    #sys.stdout.write("going back and turn ")
                    v, direction = go_back_turn(game, direction, speed)
                    rat.makeAction(game, v, training=False)
                    v = move(game, direction, speed)
                    rat.makeAction(game, v, training=False)
        
        pos = [game.get_game_variable(POSITION_X), game.get_game_variable(POSITION_Y)]
        traj.append(pos)
        #print(ori_pos, Agc, p)
        
        if pos[0]>rewardPos[0]-tolerance and pos[0]<rewardPos[0]+tolerance and pos[1]>rewardPos[1]-tolerance and pos[1]<rewardPos[1]+tolerance: 
            print("reward!", len(traj))
            rewarded = True
            rat.gc.forget()
            rat.gc.memorizeReward(rat, tRange=10, decay=0.9)
            #rat.gc.memorizeReward(rat, tRange=5, decay=0.1)
            break
        else:
            rat.gc.forget(1.01) # gradually forget
            p = np.exp(-rat.gc.gc_max)
            threshold_low = rat.gc.gc_max*0.7
            threshold_high = rat.gc.gc_max*0.95
        
        if len(traj) > 500:
            print("fail!")
            break
    plot_traj(np.array(traj), fname="unified"+trial)






### main ###

game = create_environment()
game.new_episode()
game.make_action([0, 0, 0, 0, 1], 10)
game.make_action([0, 0, 0, 0, 0], 100)

pos = [0,0]
#env_range = [[-190, 190], [-190, 190]]
traj2 = []

rat = System(lookAround(game), nSteps=5)
data = sio.loadmat('data2.mat', appendmat=False)
rat.vpc.w = data['vpc']
rat.mpc.w = data['mpc']
rat.cpc.obj.w = data['cpc']
#rat.rw = data['rw']
for i in range(5):
    rat.mgc.sheets[i].alpha = data['mgc'][i][0]
    rat.mgc.sheets[i].beta = data['mgc'][i][1]
    rat.mgc.sheets[i].A = data['mgc'][i][2].reshape((-1))
    rat.mgc.sheets[i].Wcpc = data['mgc'][i][3]
    

# experiments
nExps = 3 # number of changing reward position
nTrails = 3 # number of testing of the same reward position
tolerance = 15
env_range = [[-130, 130], [-130, 130]]
 
traj2 = []
for i in range(2000):
    if i%10 == 0:
        sys.stdout.write(str(i)+' ')
        sys.stdout.flush()
    v, direction = randMove(game, True, random.randint(3,6))
    rat.makeAction(game, v, training=False, trainHop=True, recording=True)
    traj2.append([game.get_game_variable(POSITION_X), game.get_game_variable(POSITION_Y)])

search(game, rat, [0,0], 10)
rat.gc.data = []
traj3 = []
for i in range(1000):
    if i%10 == 0:
        sys.stdout.write(str(i)+' ')
        sys.stdout.flush()
    v, direction = randMove(game, True, random.randint(3,6))
    rat.makeAction(game, v, training=False, trainHop=True, recording=True)
    rat.gc.response(rat.Acpc, record=True) 
    traj3.append([game.get_game_variable(POSITION_X), game.get_game_variable(POSITION_Y)])
    traj2.append([game.get_game_variable(POSITION_X), game.get_game_variable(POSITION_Y)])

rat.showRW()
plot_receptive_field(np.array([rat.gc.data]), np.array(traj3), plot_size=[1,1], bin_length=int(len(traj3)/15), fig_size=[5,5], fname="goal")
plot_traj(np.array(traj3), fname="goal_sim")

if noisy:
    traj2=np.array(traj2)
    plot_receptive_field(rat.mpc.data.T, traj2, plot_size=[15,15], bin_length=int(traj2.shape[0]/15), fig_size=[15,15], fname="mpc")
    plot_receptive_field(rat.vpc.data.T, traj2, plot_size=[11,11], bin_length=int(traj2.shape[0]/15), fig_size=[11,11], fname="vpc")
    plot_receptive_field(rat.cpc.obj.data.T, traj2, plot_size=[14,14], bin_length=int(traj2.shape[0]/15), fig_size=[14,14], fname="cpc")
    plot_receptive_field(rat.mgc.data.T, np.array([[a,a,a,a,a] for a in traj2]).reshape((-1, 2)), plot_size=[30,15], bin_length=int(traj2.shape[0]*5/15), fig_size=[15,30], fname="mgc")
    nExps=0

rat.gc.w = np.zeros(rat.gc.nPC)
rat.gc.gc_max = 0
for exp in range(nExps):
    rewardPos = [random.random() * (env_range[0][1]-env_range[0][0]) + env_range[0][0], random.random() * (env_range[1][1]-env_range[1][0]) + env_range[1][0]]
    print("new reward position:", rewardPos)
    for trial in range(nTrails):
        initPos(game, rat)
        #sim_gc(gc, MPCpop, [[50, 150], [50, 150]], 1000, bin_length=40) #[[50, 150], [50, 150]]
        #ratGo(game, rat, rewardPos, tolerance, str(exp+1)+"_"+str(trial+1))
        unifiedStrategy(game, rat, rewardPos, tolerance, str(exp+1)+"_"+str(trial+1))
        
game.close()



# save data into .mat file
#n=4
#simData = {'pos': trajectory, 'v1': v1.data, 'vn_height': v1.vn_height, 'vn_width': v1.vn_width}
#sio.savemat('data/v1data'+str(n)+'.mat', simData, appendmat=False)

