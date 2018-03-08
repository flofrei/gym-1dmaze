"""
World Setup Toy Example

ASCII based features and impl.
"""

import numpy as np

import gym
from gym import error, spaces, utils, core
from gym.utils import seeding
from gym.envs.registration import register
import math


class SimpleMaze(gym.Env):

    """
    # Set this in SOME subclasses
    metadata = {'render.modes': []}
    reward_range = (-np.inf, np.inf)
    spec = None


    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    """

    metadata = {'render.modes': ["human",'ansi'] }
    #reward_range = (0,2)
    action_list = ['left','right','leftskip','rightskip']

    def __init__(self,wsize=None,expansion_flag=None):

        if expansion_flag==True:
            self.action_space = spaces.Discrete(4);
            self.n_actions=4;
        elif expansion_flag==False:
            self.action_space = spaces.Discrete(2);
            self.n_actions=2;
        else:
            raise NotImplementedError

        self.world_size=wsize;
        #low = np.zeros
        #print(self.world_size)
        #self.observation_space = spaces.Discrete(self.world_size);
        self.observation_space = spaces.Box(low=0,high=2,shape=(self.world_size,),dtype=np.int8)

        self.world = None;
        self.viewer = None;
        #print(self.observation_space.shape);
        #self.action_space = spaces.Box(low=-10, high=10, shape=(1,))

        #if self.world_size > 0:
        #    self.observation_space = spaces.Discrete(self.world_size);
        #else:
        #    raise NotImplementedError
        
        self.agent_position=-1;
        self.goal_position=-1;

        #self.world=self.create_world();
        #self.set_startposition(3);
        #self.set_goalposition(14);

    def create_world(self):
        #listholder = [];
        #for k in range(self.world_size):
        #    listholder+=['_'];
        #   
        #return listholder;
        inital = np.zeros(self.world_size);
        self.world = inital;

    def swap(self,pos1,pos2):
            self.world[pos1],self.world[pos2] = self.world[pos2],self.world[pos1];

    def set_startposition(self,position):
        self.world[position] = 1;
        self.agent_position=position;

    def set_goalposition(self,position):
        self.world[position] = 2;
        self.goal_position=position;

    def _reset(self):
        #self.world=self.create_world();
        inital = np.zeros(self.world_size);
        self.world = inital;
        self.world[3]=1;
        self.world[14]=2;
        self.agent_position=3;
        self.goal_position=14;
        #self.set_startposition(3);
        #self.set_goalposition(14);
        #warray = self.get_features();
        return self.world;

    def new_print(self):
        #size = len(world);
        #newstring = ''.join(self.world);
        print(self.world);

    def get_features(self):
        #new_input = np.zeros(self.world_size); 
        #for r in range(self.world_size):
        #    new_input[r] = ord(self.world[r]);

        return self.world

    def _render(self, mode='ansi',close=False):
        if mode == 'ansi':
            newstring = str(self.world);
            #print(newstring);
            return newstring;
        else:
            super(SimpleMaze, self).render(mode=mode) # just raise an exception

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _close(self):
        del action_space;

    """Run one timestep of the environment's dynamics. When end of
    episode is reached, you are responsible for calling `reset()`
    to reset this environment's state.

    Accepts an action and returns a tuple (observation, reward, done, info).

    Args:
        action (object): an action provided by the environment

    Returns:
        observation (object): agent's observation of the current environment
        reward (float) : amount of reward returned after previous action
        done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
        info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
    """
    def _step(self, action_value):
        #Step function in the 1D World
        #print("------------step----------------")

        if(action_value==0 or action_value==1):
            P=1;
            #print("everything is alrigth")
        else:
            print("Something terribly went wrong")

        action = self.action_list[action_value];
        ## First action value
        if action == 'left':
            ## Going out of the world case
            if self.agent_position==0:
                r = 0; 
                end_of_eps = False;
                lst = {};
                return self.get_features(),r,end_of_eps,lst;
            ## Moving internally in the world
            else:
                ## Moving to goal position 
                if self.agent_position==self.goal_position+1: 
                    r = 1;
                    tstate = self._reset();
                    end_of_eps = True;
                    lst = {};
                    return tstate, r, end_of_eps, lst;
                ## only moving around  
                else:
                    r = 0;
                    end_of_eps = False;
                    lst = {};
                    self.swap(self.agent_position,self.agent_position-1);
                    self.agent_position=self.agent_position-1;
                    return self.get_features(),r,end_of_eps,lst;
        ## Second action value        
        elif action == 'right':
            ## Going out of the world case
            if self.agent_position==self.world_size-1:
                r = 0;
                end_of_eps = False;
                lst = {};
                return self.get_features(),r,end_of_eps,lst;
            ## Moving internally in the World
            else:
                ## Moving to goal position
                if self.agent_position==self.goal_position-1:
                    r = 1;
                    tstate = self._reset();
                    end_of_eps = True;
                    lst = {};
                    return tstate, r, end_of_eps, lst;
                ## only moving around
                else:
                    r = 0;
                    end_of_eps = False;
                    lst = {};
                    self.swap(self.agent_position,self.agent_position+1);
                    self.agent_position=self.agent_position+1;
                    return self.get_features(), r,end_of_eps,lst;
        ## Third action value  
        elif action == 'leftskip':
            ## Goind out of the world
            if self.agent_position==1 or self.agent_position==0:
                r = 0;
                end_of_eps = False;
                lst = {};
                return self.get_features(), r,end_of_eps,lst;
            ## Moving internally in the world
            else:
                ## Moving to goal position
                if self.agent_position == self.goal_position+2:
                    r = 1;
                    tstate = self._reset();
                    end_of_eps = True;
                    lst = {};
                    return tstate, r,end_of_eps,lst;
                ## only moving around
                else:
                    r = 0;
                    end_of_eps = False;
                    lst = {};
                    self.swap(self.agent_position,self.agent_position-2);
                    self.agent_position=self.agent_position-2;
                    return self.get_features(), r,end_of_eps,lst;
        ## Fourth action value 
        elif action == 'rightskip':
            ## Goind out of the world
            if self.agent_position==self.world_size-1 or self.agent_position==self.world_size-2:
                r = 0;
                end_of_eps = False;
                lst = {};
                return self.get_features(), r,end_of_eps,lst;
            ## Moving internally in the world
            else:
                ## Moving to goal position
                if self.agent_position == self.goal_position-2:
                    r = 1;
                    tstate = self._reset();
                    end_of_eps = True;
                    lst = {};
                    return tstate, r,end_of_eps,lst;
                ## only moving around
                else:
                    r = 0;
                    end_of_eps = False;
                    lst = {};
                    self.swap(self.agent_position,self.agent_position+2);
                    self.agent_position=self.agent_position+2;
                    return self.get_features(), r,end_of_eps,lst;

"""
def test_suite():
    size = 16;
    env = SimpleWorld(size);
    print("First tests with moving right");
    env.set_startposition(3);
    env.set_goalposition(15);
    #print(new_world);
    env.new_print(); 
    print(env.get_features());
    
    for k in range(12):
        action = 'right'; 
        newstate,reward = env.step(action);
        env.new_print(); print(reward)
        #print("Total score for {} is {}".format(newstate, reward))

    env.reset();
    env.set_startposition(3);
    env.set_goalposition(15);

    for k in range(6):
        action = 'rightskip'; 
        newstate,reward = env.step(action);
        env.new_print(); print(reward)

    print("Second tests with moving left");
    env.reset();
    env.set_startposition(15);
    env.set_goalposition(3);
    #print(new_world);
    env.new_print(); 
    print(env.get_features());
    
    for k in range(12):
        action = 'left';
        newstate,reward = env.step(action);
        env.new_print(); print(reward)
        #print("Total score for {} is {}".format(newstate, reward))

    env.reset();
    env.set_startposition(15);
    env.set_goalposition(3);
    for k in range(6):
        action = 'leftskip'; 
        newstate,reward = env.step(action);
        env.new_print(); print(reward)
"""

class SimpleMaze1x16s(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x16s, self).__init__(wsize=16,expansion_flag=False)

class SimpleMaze1x16c(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x16c, self).__init__(wsize=16,expansion_flag=True)

    
