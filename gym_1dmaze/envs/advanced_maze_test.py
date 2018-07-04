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
from copy import copy, deepcopy

class AdvancedMazeLine(gym.Env):
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
    action_list = ['left','right','up','down']

    def __init__(self,rows=None,columns=None,wmode=None):
        self.action_space = spaces.Discrete(4)
        self.n_actions=4;
        self.world_number_of_rows=rows;
        self.world_number_of_columns=columns;
        self.world_mode = wmode;
        self.number_of_episodes=0;
        self.number_of_steps_taken_in_episode=0;

        self.agent_position=[-1,-1];
        self.goal_row = -1
        self.goal_column = -1
        print("Created instance!")

        self.observation_space = spaces.Box(low=0,high=2,shape=(self.world_number_of_rows,self.world_number_of_columns),dtype=np.int8)
        self.world = None;
        self.world_as_string = None;
        self.blank_string = None;
        self.empty_string()
        self.right_decision = {};

    def empty_string(self):
        row_holder = []
        tmp = []
        for j in range(self.world_number_of_columns):
            tmp.append('_')

        for k in range(self.world_number_of_rows):
            row_holder.append( copy(tmp) )
        self.blank_string=row_holder

    def empty_world(self):
        row_holder = []
        row_ = np.zeros(self.world_number_of_columns)
        for k in range(self.world_number_of_rows):
            row_holder.append( copy(row_) )
        self.world = np.array(row_holder)

    def to_liststring(self):
        self.world_as_string = deepcopy(self.blank_string)
        self.string_setter_line(self.goal_row,self.goal_column,'T')
        self.string_setter(self.agent_position,'a')

    def concat_string(self):
        new_rows = []
        for k in range(self.world_number_of_rows):
            row_k = copy( self.world_as_string[k] )
            new_row = ''.join(row_k) #not string instance
            new_row += '\n'
            new_rows.append(new_row)

        new_world = ''.join(new_rows) 
        return new_world

    def string_setter(self,position,str_val):
        world_row_ref = self.world_as_string[position[0]]
        world_row_ref[position[1]] = str_val

    def string_setter_line(self,row_ind,column_ind,str_val):
        if(row_ind==-1):
            for k in range(self.world_number_of_rows):
                row_k = self.world_as_string[k]
                row_k[column_ind] = str_val
        if(column_ind==-1):
            row_ref = self.world_as_string[row_ind]
            for k in range(self.world_number_of_columns):
                row_ref[k]=str_val

    def mover(self,rel_movment,symb):
        #new_agent_pos = [copy(self.agent_position[0])+rel_movment[0], copy(self.agent_position[1])+rel_movment[1]]
        new_agent_pos = copy( [sum(x) for x in zip(self.agent_position, rel_movment)] )
        self.move_agent(new_agent_pos,symb)

    def move_agent(self,new_position,symb):
        old_position = self.agent_position
        self.world[old_position[0],old_position[1]] = symb
        self.world[new_position[0],new_position[1]] = 1.
        self.agent_position = new_position
    
    def fill_line(self):
        if(self.goal_row == -1):
            for k in range(self.world_number_of_rows):
                self.world[k,self.goal_column]=2.
        if(self.goal_column == -1):
            for k in range(self.world_number_of_columns):
                self.world[self.goal_row,k]=2.
    def fill_agent(self):
        self.world[self.agent_position[0],self.agent_position[1]] = 1.

    def _reset(self):
        self.empty_world()
        self.number_of_steps_taken_in_episode=0;
        self.agent_position = [1,1]
        self.goal_row = self.world_number_of_rows-1
        self.top=False
        self.bottom=True
        self.left=False
        self.right=False
        goal_ind = 3
        self.goal_column = -1

        if self.world_mode == 'mode42':
            goal=-1
            goal_ind = np.random.randint(low=0,high=4)
            pos1 = np.random.randint(low=1,high=self.world_number_of_rows-1)          
            pos2 = np.random.randint(low=1,high=self.world_number_of_columns-1)
            agent_pos = [pos1,pos2]
            if(goal_ind==0):
                self.goal_row = -1
                self.goal_column = 0
                self.top=False
                self.bottom=False
                self.left=True
                self.right=False
            elif(goal_ind==1):
                self.goal_row = -1
                self.goal_column = self.world_number_of_columns-1
                self.top=False
                self.bottom=False
                self.left=False
                self.right=True
            elif(goal_ind==2):
                self.goal_row = 0
                self.goal_column = -1
                self.top=True
                self.bottom=False
                self.left=False
                self.right=False
            elif(goal_ind==3):
                self.goal_row = self.world_number_of_rows-1
                self.goal_column = -1
                self.top=False
                self.bottom=True
                self.left=False
                self.right=False
            else:
                print("Bad branching")
        
        self.fill_line()
        self.fill_agent()

        if(goal_ind == 0):
            self.right_decision['left']=1.
        elif(goal_ind == 1):
            self.right_decision['right']=1.
        elif(goal_ind == 2):
            self.right_decision['up']=1.
        elif(goal_ind == 3):
            self.right_decision['down']=1.

        self.number_of_episodes +=1 ;
        return np.copy(self.world),copy(self.right_decision);


    def _render(self, mode='human',close=False):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        self.to_liststring();
        out_string = self.concat_string()

        outfile.write(out_string);
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

        left_border=False
        right_border=False
        bottom_border=False
        top_border=False
        going_up=None
        going_down=None
        going_left=None
        going_right=None

        if(self.agent_position[0]==0):
            top_border=True
            if(action == 'up'):
                going_up = True
            else:
                going_up = False
        if(self.agent_position[0]==self.world_number_of_rows-1):
            bottom_border=True
            if(action == 'down'):
                going_down = True
            else:
                going_down = False
        if(self.agent_position[1]==0):
            left_border=True
            if(action == 'left'):
                going_left = True
            else:
                going_left = False
        if(self.agent_position[1]==self.world_number_of_columns-1):
            right_border=True
            if(action == 'right'):
                going_right = True
            else:
                going_right = False

        
        if action == 'left':
            rel_mov = [0,-1]
        elif action == 'right':
            rel_mov = [0,1]
        elif action == 'up':
            rel_mov = [-1,0]
        elif action == 'down':
            rel_mov = [1,0]
    
        if(going_up and self.top):
            return self._automatic_end_returner();
        elif(going_down and self.bottom):
            return self._automatic_end_returner();
        elif(going_left and self.left):
            return self._automatic_end_returner();
        elif(going_right and self.right):
            return self._automatic_end_returner();
        else:
            if(going_up and top_border):
                return self._automatic_wall_returner();
            elif(going_down and bottom_border):
                return self._automatic_wall_returner();
            elif(going_left and left_border):
                return self._automatic_wall_returner();
            elif(going_right and right_border):
                return self._automatic_wall_returner();
            else:
                #move
                if(self.top and top_border):
                    symb = 2.
                elif(self.bottom and bottom_border):
                    symb = 2.
                elif(self.left and left_border):
                    symb = 2.
                elif(self.right and right_border):
                    symb = 2.
                else:
                    symb = 0.
                self.mover(rel_mov,symb)
                return self._automatic_step_returner();

    def _automatic_end_returner(self):
        r = 1;
        end_of_eps = True;
        lst = {self.number_of_steps_taken_in_episode};
        return np.copy(self.world),r,end_of_eps,lst;

    def _automatic_step_returner(self):
        r = 0;
        end_of_eps = False;
        lst = {-1};
        return np.copy(self.world),r,end_of_eps,lst;

    def _automatic_wall_returner(self):
        r = -1;
        end_of_eps = True;
        lst = {-1};
        return np.copy(self.world),r,end_of_eps,lst;

class AdvancedMazeLine4x4m42(AdvancedMazeLine):

    def __init__(self):
        super(AdvancedMazeLine4x4m42, self).__init__(rows=4,columns=4,wmode='mode42')

class AdvancedMazeLine4x4m0(AdvancedMazeLine):

    def __init__(self):
        super(AdvancedMazeLine4x4m0, self).__init__(rows=4,columns=4,wmode='mode0')

class AdvancedMazeLine10x10m42(AdvancedMazeLine):

    def __init__(self):
        super(AdvancedMazeLine10x10m42, self).__init__(rows=10,columns=10,wmode='mode42')

class AdvancedMazeLine10x10m0(AdvancedMazeLine):

    def __init__(self):
        super(AdvancedMazeLine10x10m0, self).__init__(rows=10,columns=10,wmode='mode0')

