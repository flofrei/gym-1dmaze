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
    

    def __init__(self,wsize=None,extended_action_set=None,wmode=None,circular_world=None):

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
        self.number_of_steps_taken_in_episode=0;
        self.circular = circular_world;
        self.number_of_minimal_actions=-1;
        
        self.positionset1 = [0 , self.world_size -1];
        self.positionset2 = [self.world_size -2 , 3 ];
        self.positionset3 = [ int(self.world_size/2), self.world_size-1];
        self.positionset4 = [ self.world_size-1 , int(self.world_size/2)];
        

        self.agent_position=-1;
        self.goal_position=-1;

        if self.world_mode == 'mode0':
            self.observation_space = spaces.Box(low=0,high=2,shape=(self.world_size,),dtype=np.int8)
        elif self.world_mode == 'mode10':
        	self.observation_space = spaces.Box(low=0,high=2,shape=(self.world_size,),dtype=np.int8)
        elif self.world_mode == 'mode1':
            self.observation_space = spaces.Box(low=0,high=3,shape=(self.world_size,),dtype=np.int8)
            self.treasure_position=-1;
        elif self.world_mode == 'mode42':
        	self.observation_space = spaces.Box(low=0,high=2,shape=(self.world_size,),dtype=np.int8)
        else:
            print("Something has terribly gone wrong!");

        self.world = None;
        self.world_as_string = None;

    def to_liststring(self):
        listholder = [];
        for k in range(self.world_size):
            listholder+=['_'];

        listholder[self.agent_position] = 'a';
        listholder[self.goal_position] = 'T';

        if self.world_mode == 'mode1':
            listholder[self.treasure_position] = 't';
        
        return listholder;

    def swap(self,pos1,pos2):
            self.world[pos1],self.world[pos2] = self.world[pos2],self.world[pos1];

    def _reset(self):
        inital = np.zeros(self.world_size);
        self.number_of_steps_taken_in_episode=0;
        startpos, goalpos = self.positionset1;

        if self.world_mode == 'mode0':
            startpos, goalpos = self.positionset1;

        if self.world_mode == 'mode10':
            p = np.random.randint(0,2)
            if p == 0:
                startpos,goalpos = self.positionset1;
            else:
                startpos,goalpos = self.positionset2;

        if self.world_mode == 'mode42':
            pos1=0
            pos2=0
            while pos1 == pos2:
                pos1 = np.random.randint(0,self.world_size)
                pos2 = np.random.randint(0,self.world_size)
            startpos,goalpos = pos1,pos2        
        
        self.world = inital;
        self.world[startpos]=1;
        self.world[goalpos]=2;

        self.agent_position=startpos;
        self.goal_position=goalpos;

        if(self.circular):
            if(startpos>goalpos):
                diff = startpos - goalpos;
            else:
                diff = goalpos - startpos;
            self.number_of_minimal_actions = min(diff,self.world_size-diff)
        else:
            if(startpos>goalpos):
                self.number_of_minimal_actions = startpos - goalpos;
            else:
                self.number_of_minimal_actions = goalpos - startpos;

        self.number_of_episodes +=1 ;
        return self.world;


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

        action = self.action_list[action_value];
        self.number_of_steps_taken_in_episode +=1;
        ## First action value
        if action == 'left':
            ## Going out of the world case
            if self.agent_position==0:
                if(self.circular):
                    if(self.goal_position==self.world_size-1):
                        return self._automatic_end_returner();
                    else:
                        self.swap(self.agent_position,self.world_size-1);
                        self.agent_position=self.world_size-1;
                        return self._automatic_step_returner();
                else:
                    return self._automatic_step_returner();          
            ## Moving internally in the world
            else:
                ## Moving to goal position 
                if self.agent_position==self.goal_position+1:
                    return self._automatic_end_returner();
                ## only moving around  
                else:
                    self.swap(self.agent_position,self.agent_position-1);
                    self.agent_position=self.agent_position-1;
                    return self._automatic_step_returner();
        ## Second action value        
        elif action == 'right':
            ## Going out of the world case
            if self.agent_position==self.world_size-1:
                if(self.circular):
                    if(self.goal_position==0):
                        return self._automatic_end_returner();
                    else:
                        self.swap(self.agent_position,0);
                        self.agent_position=0;
                        return self._automatic_step_returner();
                else:
                    return self._automatic_step_returner();
            ## Moving internally in the World
            else:
                ## Moving to goal position
                if self.agent_position==self.goal_position-1:
                    return self._automatic_end_returner();
                ## only moving around
                else:
                    self.swap(self.agent_position,self.agent_position+1);
                    self.agent_position=self.agent_position+1;
                    return self._automatic_step_returner()
        ## Third action value  
        elif action == 'leftskip':
            ## Goind out of the world
            if self.agent_position==1 or self.agent_position==0:
                if(self.circular):
                    if(self.agent_position==1):
                        if(self.goal_position==self.world_size-1):
                            return self._automatic_end_returner();
                        else:
                            self.swap(self.agent_position,self.world_size-1);
                            self.agent_position = self.world_size-1;
                            return self._automatic_step_returner();
                    if(self.agent_position==0):
                        if(self.goal_position==self.world_size-2):
                            return self._automatic_end_returner();
                        else:
                            self.swap(self.agent_position,self.world_size-2);
                            self.agent_position = self.world_size-2;
                            return self._automatic_step_returner();
                else:
                    return self._automatic_step_returner();
            ## Moving internally in the world
            else:
                ## Moving to goal position
                if self.agent_position == self.goal_position+2:
                    return self._automatic_end_returner();
                ## only moving around
                else:
                    self.swap(self.agent_position,self.agent_position-2);
                    self.agent_position=self.agent_position-2;
                    return self._automatic_step_returner();
        ## Fourth action value 
        elif action == 'rightskip':
            ## Goind out of the world
            if self.agent_position==self.world_size-1 or self.agent_position==self.world_size-2:
                if(self.circular):
                    if(self.agent_position==self.world_size-1):
                        if(self.goal_position==1):
                            return self._automatic_end_returner();
                        else:
                            self.swap(self.agent_position,1);
                            self.agent_position=1;
                            return _automatic_step_returner();
                    if(self.agent_position==self.world_size-2):
                        if(self.goal_position==0):
                            return self._automatic_end_returner();
                        else:
                            self.swap(self.agent_position,0);
                            self.agent_position=0;
                            return _automatic_step_returner();
                else:
                    return self._automatic_step_returner();
            ## Moving internally in the world
            else:
                ## Moving to goal position
                if self.agent_position == self.goal_position-2:
                    return self._automatic_end_returner();
                ## only moving around
                else:
                    self.swap(self.agent_position,self.agent_position+2);
                    self.agent_position=self.agent_position+2;
                    return self._automatic_step_returner();

    def _automatic_end_returner(self):
        r = 1;
        end_of_eps = True;
        lst = {self.number_of_steps_taken_in_episode};
        return self.world,r,end_of_eps,lst;

    def _automatic_step_returner(self):
        #if(self.number_of_steps_taken_in_episode<=self.number_of_minimal_actions):
        #    r = 0;
        #else:
        #    r = -0.1;
        r = 0;
        end_of_eps = False;
        lst = {-1};
        return self.world,r,end_of_eps,lst;

    def _automatic_overstep_returner(self):
        r = 0;
        end_of_eps = True;
        lst = {};
        return self.world,r,end_of_eps,lst;


class SimpleMaze1x16sasm0cw0(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x16sasm0cw0, self).__init__(wsize=16,extended_action_set=False,wmode='mode0',circular_world=False)

class SimpleMaze1x16easm0cw0(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x16easm0cw0, self).__init__(wsize=16,extended_action_set=True,wmode='mode0',circular_world=False)

class SimpleMaze1x32sasm0cw0(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x32sasm0cw0, self).__init__(wsize=32,extended_action_set=False,wmode='mode0',circular_world=False)

class SimpleMaze1x32easm0cw0(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x32easm0cw0, self).__init__(wsize=32,extended_action_set=True,wmode='mode0',circular_world=False)

class SimpleMaze1x16sasm10cw0(SimpleMaze):

	def __init__(self):
		super(SimpleMaze1x16sasm10cw0,self).__init__(wsize=16,extended_action_set=False,wmode='mode10',circular_world=False)
    
class SimpleMaze1x16easm10cw0(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x16easm10cw0, self).__init__(wsize=16,extended_action_set=True,wmode='mode10',circular_world=False)

class SimpleMaze1x32sasm10cw0(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x32sasm10cw0, self).__init__(wsize=32,extended_action_set=False,wmode='mode10',circular_world=False)

class SimpleMaze1x32easm10cw0(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x32easm42cw0, self).__init__(wsize=32,extended_action_set=True,wmode='mode10',circular_world=False)

class SimpleMaze1x16sasm42cw0(SimpleMaze):

	def __init__(self):
		super(SimpleMaze1x16sasm42cw0,self).__init__(wsize=16,extended_action_set=False,wmode='mode42',circular_world=False)
    
class SimpleMaze1x16easm42cw0(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x16easm42cw0, self).__init__(wsize=16,extended_action_set=True,wmode='mode42',circular_world=False)

class SimpleMaze1x32sasm42cw0(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x32sasm42cw0, self).__init__(wsize=32,extended_action_set=False,wmode='mode42',circular_world=False)

class SimpleMaze1x32easm42cw0(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x32easm42cw0, self).__init__(wsize=32,extended_action_set=True,wmode='mode42',circular_world=False)

class SimpleMaze1x16sasm42cw1(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x16sasm42cw1,self).__init__(wsize=16,extended_action_set=False,wmode='mode42',circular_world=True)
    
class SimpleMaze1x16easm42cw1(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x16easm42cw1, self).__init__(wsize=16,extended_action_set=True,wmode='mode42',circular_world=True)

class SimpleMaze1x32sasm42cw1(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x32sasm42cw1, self).__init__(wsize=32,extended_action_set=False,wmode='mode42',circular_world=True)

class SimpleMaze1x32easm42cw1(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x32easm42cw1, self).__init__(wsize=32,extended_action_set=True,wmode='mode42',circular_world=True)

class SimpleMaze1x4sasm42cw0(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x4sasm42cw0,self).__init__(wsize=4,extended_action_set=False,wmode='mode42',circular_world=False)
    
class SimpleMaze1x4easm42cw0(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x4easm42cw0, self).__init__(wsize=4,extended_action_set=True,wmode='mode42',circular_world=False)

class SimpleMaze1x4sasm0cw0(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x4sasm0cw0,self).__init__(wsize=4,extended_action_set=False,wmode='mode0',circular_world=False)
    
class SimpleMaze1x4easm0cw0(SimpleMaze):

    def __init__(self):
        super(SimpleMaze1x4easm0cw0, self).__init__(wsize=4,extended_action_set=True,wmode='mode0',circular_world=False)