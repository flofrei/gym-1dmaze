"""
World Setup Toy Example

ASCII based features and impl.
"""

import numpy as np

import gym
from gym import error, spaces, utils, core
from gym.utils import seeding
from gym.envs.registration import register
from six import StringIO
import math
import sys


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

    metadata = {'render.modes': ["human", 'ansi']}
    action_list = ['left','right','leftskip','rightskip']
    

    def __init__(self,wsize=None,extended_action_set=None,wmode=None):

        if extended_action_set==True:
            self.action_space = spaces.Discrete(4);
            self.n_actions=4;
        elif extended_action_set==False:
            self.action_space = spaces.Discrete(2);
            self.n_actions=2;
        else:
            print("Something has terribly gone wrong!");

        self.world_size=wsize;
        self.world_mode = wmode;
        self.number_of_episodes=0;
        
        self.positionset1 = [3 , self.world_size -2];
        self.positionset2 = [self.world_size -2 , 3 ];
        self.positionset3 = [ int(self.world_size/2), self.world_size-1];
        self.positionset4 = [ self.world_size-1 , int(self.world_size/2)];
        

        if self.world_mode == 'mode0':
            self.observation_space = spaces.Box(low=0,high=2,shape=(self.world_size,),dtype=np.int8)
            self.agent_position=-1;
            self.goal_position=-1;
        elif self.world_mode == 'mode1':
            self.observation_space = spaces.Box(low=0,high=3,shape=(self.world_size,),dtype=np.int8)
            self.agent_position=-1;
            self.goal_position=-1;
            self.treasure_position=-1;
        else:
            print("Something has terribly gone wrong!");

        self.world = None;
        self.world_as_string = None;

    def to_liststring(self):
        listholder = [];
        for k in range(self.world_size):
            listholder+=['_'];

        if self.world_mode == 'mode0':
            listholder[self.agent_position] = 'a';
            listholder[self.goal_position] = 'T';
        elif self.world_mode == 'mode1':
            listholder[self.agent_position] = 'a';
            listholder[self.goal_position] = 'T';
            listholder[self.treasure_position] = 't';
        
        return listholder;

    def swap(self,pos1,pos2):
            self.world[pos1],self.world[pos2] = self.world[pos2],self.world[pos1];

    def _reset(self):
        inital = np.zeros(self.world_size);

        if self.world_mode == 'mode0':
            if self.number_of_episodes < 250:
                startpos, goalpos = self.positionset1;
            elif self.number_of_episodes < 500:
                startpos, goalpos = self.positionset2;
            elif self.number_of_episodes < 750:
                startpos, goalpos = self.positionset3;
            else:
                startpos, goalpos = self.positionset4;
        else:
            print("Not implemented YET");
            #TODO        
        
        self.world = inital;
        self.world[startpos]=1;
        self.world[goalpos]=2;

        self.agent_position=startpos;
        self.goal_position=goalpos;

        self.number_of_episodes +=1 ;
        return self.world;


    def get_features(self):
        return self.world


    def _render(self, mode='human',close=False):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
    
        lstholder = self.to_liststring();
        outfile.write(''.join(lstholder));
        outfile.write("\n");

        # No need to return anything for human
        if mode != 'human':
            return outfile
        
        #TODO ansi mode is not working yet since StringIO is not used correctly

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


class SimpleMaze1x16sasm0(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x16sasm0, self).__init__(wsize=16,extended_action_set=False,wmode='mode0')

class SimpleMaze1x16easm0(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x16easm0, self).__init__(wsize=16,extended_action_set=True,wmode='mode0')

class SimpleMaze1x32sasm0(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x32sasm0, self).__init__(wsize=32,extended_action_set=False,wmode='mode0')

class SimpleMaze1x32easm0(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x32easm0, self).__init__(wsize=32,extended_action_set=True,wmode='mode0')
    
