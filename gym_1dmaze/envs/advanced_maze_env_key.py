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

class AdvancedMazeKey(gym.Env):
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

        self.positionset1 = [ [1,1] ,[self.world_number_of_rows-1,self.world_number_of_columns-1] ];

        self.agent_position=[-1,-1];
        self.goal_position=[-1,-1];
        self.key_position=[-1,-1];
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

        self.string_setter(self.agent_position,'a')
        self.string_setter(self.goal_position,'T')
        self.string_setter(self.key_position,'k')

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

    def mover(self,rel_movment):
        #new_agent_pos = [copy(self.agent_position[0])+rel_movment[0], copy(self.agent_position[1])+rel_movment[1]]
        new_agent_pos = copy( [sum(x) for x in zip(self.agent_position, rel_movment)] )
        self.move_agent(new_agent_pos)

    def move_agent(self,new_position):
        old_position = self.agent_position
        self.world[old_position[0],old_position[1]] = 0.
        self.world[new_position[0],new_position[1]] = 1.
        self.agent_position = new_position
        if(not self.key_picked_up):
            self.world[self.goal_position[0],self.goal_position[1]]=2.

    def _reset(self):
        self.empty_world()
        self.number_of_steps_taken_in_episode=0;
        [startpos, goalpos] = self.positionset1;
        keypos = [0,0]
        self.key_picked_up = False

        if self.world_mode == 'mode42':
            pos1=[0,0]
            pos2=[0,0]
            while pos1 == pos2:
                new_column_inds = np.random.randint(low=0,high=self.world_number_of_columns,size=3)
                new_row_inds = np.random.randint(low=0,high=self.world_number_of_rows,size=3)
                pos1 = [new_row_inds[0],new_column_inds[0]]
                pos2 = [new_row_inds[1],new_column_inds[1]]
                pos3 = [new_row_inds[2],new_column_inds[2]]
            startpos = pos1
            goalpos = pos2
            while(pos3==pos1 or pos3==pos2):
                pos3 = [np.random.randint(low=0,high=self.world_number_of_rows),np.random.randint(low=0,high=self.world_number_of_columns)]
            keypos = pos3

        if self.world_mode == 'mode20':
            startpos=[2,3]
            goalpos=[2,0]
        if self.world_mode == 'mode21':
            startpos=[2,0]
            goalpos=[2,3]
        if self.world_mode == 'mode22':
            startpos=[3,2]
            goalpos=[0,2]
        if self.world_mode == 'mode23':
            startpos=[0,2]
            goalpos=[3,2]

        if self.world_mode == 'mode29':
            nmb = np.random.randint(low=0,high=4)
            if(nmb == 0):
                startpos=[2,3]
                goalpos=[2,0]
            elif(nmb == 1):
                startpos=[2,0]
                goalpos=[2,3]
            elif(nmb == 2):
                startpos=[3,2]
                goalpos=[0,2]
            else:
                startpos=[0,2]
                goalpos=[3,2]

        if self.world_mode == 'mode33':
            rng = np.random.rand()
            modal = False
            if(rng>0.5):
                row = np.random.randint(low=0,high=self.world_number_of_rows)
                modal = True
            else:
                column = np.random.randint(low=0,high=self.world_number_of_columns)
                
            if(modal):
                #same row need 2 position on the same row
                column_ind = np.random.randint(low=0,high=self.world_number_of_columns,size=2)
                col_ind_1 = column_ind[0]
                col_ind_2 = column_ind[1]
                while(col_ind_1==col_ind_2):
                    col_ind_2 = np.random.randint(low=0,high=self.world_number_of_columns)
                startpos = [row,col_ind_1]
                goalpos = [row,col_ind_2]
            else:
                #same column need 2 positions on the same column
                row_ind = np.random.randint(low=0,high=self.world_number_of_rows,size=2)
                row_ind_1 = row_ind[0]
                row_ind_2 = row_ind[1]
                while(row_ind_1==row_ind_2):
                    row_ind_2 = np.random.randint(low=0,high=self.world_number_of_rows)
                startpos = [row_ind_1,column]
                goalpos = [row_ind_2,column]

        self.world[startpos[0],startpos[1]]=1.
        self.world[goalpos[0],goalpos[1]]=2.
        self.world[keypos[0],keypos[1]]=3.

        self.agent_position=startpos
        self.goal_position=goalpos
        self.key_position=keypos

        if(startpos[0]>goalpos[0]):
            a = startpos[0] - goalpos[0];
            self.right_decision['left']=a
        else:
            a = goalpos[0] - startpos[0];
            self.right_decision['right']=a

        if(startpos[1]>goalpos[1]):
            b = startpos[1] - goalpos[1];
            self.right_decision['down']=b

        else:
            b = goalpos[1] - startpos[1];
            self.right_decision['up']=b

        self.number_of_minimal_actions=a+b

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
        #Step function in the 2D World
        action = self.action_list[action_value];
        self.number_of_steps_taken_in_episode +=1;

        left_border=False
        right_border=False
        bottom_border=False
        top_border=False

        if(self.agent_position[0]==0):
            top_border=True
        if(self.agent_position[0]==self.world_number_of_rows-1):
            bottom_border=True
        if(self.agent_position[1]==0):
            left_border=True
        if(self.agent_position[1]==self.world_number_of_columns-1):
            right_border=True

        ## First action value
        if action == 'left':
            rel_mov = [0,-1]           
            goal_flag = ([sum(x) for x in zip(self.agent_position, rel_mov)] == self.goal_position)
            key_flag = ([sum(x) for x in zip(self.agent_position, rel_mov)] == self.key_position)
            ## Going out of the world case
            if(left_border):
                return self._automatic_wall_returner();
            ## Moving internally in the world
            else:
                ## Moving to goal position
                if(goal_flag and self.key_picked_up):
                    return self._automatic_end_returner();
                elif(key_flag and not self.key_picked_up):
                    self.key_picked_up = True
                    self.world[self.key_position[0],self.key_position[1]]=0.
                    self.mover(rel_mov)
                    return self._automatic_key_returner();
                ## only moving around
                else:
                    self.mover(rel_mov)
                    return self._automatic_step_returner();
        ## Second action value
        elif action == 'right':
            rel_mov = [0,1]
            goal_flag = ([sum(x) for x in zip(self.agent_position, rel_mov)] == self.goal_position)
            key_flag = ([sum(x) for x in zip(self.agent_position, rel_mov)] == self.key_position)
            ## Going out of the world case
            if(right_border):
                return self._automatic_wall_returner();
            ## Moving internally in the World
            else:
                ## Moving to goal position
                if(goal_flag and self.key_picked_up):
                    return self._automatic_end_returner();
                elif(key_flag and not self.key_picked_up):
                    self.key_picked_up = True
                    self.world[self.key_position[0],self.key_position[1]]=0.
                    self.mover(rel_mov)
                    return self._automatic_key_returner();
                ## only moving around
                else:
                    self.mover(rel_mov)
                    return self._automatic_step_returner()
        ## Third action value
        elif action == 'up':
            rel_mov = [-1,0]
            goal_flag = ([sum(x) for x in zip(self.agent_position, rel_mov)] == self.goal_position)
            key_flag = ([sum(x) for x in zip(self.agent_position, rel_mov)] == self.key_position)
            ## Going out of the world case
            if(top_border):
                return self._automatic_wall_returner();
            ## Moving internally in the world
            else:
                ## Moving to goal position
                if(goal_flag and self.key_picked_up):
                    return self._automatic_end_returner();
                elif(key_flag and not self.key_picked_up):
                    self.key_picked_up = True
                    self.world[self.key_position[0],self.key_position[1]]=0.
                    self.mover(rel_mov)
                    return self._automatic_key_returner();
                ## only moving around
                else:
                    self.mover(rel_mov)
                    return self._automatic_step_returner();
        ## Forth action value
        elif action == 'down': 
            rel_mov = [1,0]
            goal_flag = ([sum(x) for x in zip(self.agent_position, rel_mov)] == self.goal_position)
            key_flag = ([sum(x) for x in zip(self.agent_position, rel_mov)] == self.key_position)
            ## Going out of the world case
            if(bottom_border):
                return self._automatic_wall_returner();
            ## Moving internally in the world
            else:
                ## Moving to goal position
                if(goal_flag and self.key_picked_up):
                    return self._automatic_end_returner();
                elif(key_flag and not self.key_picked_up):
                    self.key_picked_up = True
                    self.world[self.key_position[0],self.key_position[1]]=0.
                    self.mover(rel_mov)
                    return self._automatic_key_returner();
                ## only moving around
                else:
                    self.mover(rel_mov)
                    return self._automatic_step_returner();

    def _automatic_end_returner(self):
        r = 1.;
        #r = 0.;
        end_of_eps = True;
        #end_of_eps = False;
        lst = {self.number_of_steps_taken_in_episode,self.key_picked_up};
        #lst = {-1};
        return np.copy(self.world),r,end_of_eps,lst;

    def _automatic_step_returner(self):
        r = 0;
        end_of_eps = False;
        lst = {-1,self.key_picked_up};
        return np.copy(self.world),r,end_of_eps,lst;

    def _automatic_key_returner(self):
        r = 0.;
        end_of_eps = False;
        #end_of_eps = True;
        lst = {-1,self.key_picked_up};
        #lst = {self.number_of_steps_taken_in_episode};
        return np.copy(self.world),r,end_of_eps,lst;

    def _automatic_wall_returner(self):
        r = -1.;
        end_of_eps = True;
        lst = {-1,self.key_picked_up};
        return np.copy(self.world),r,end_of_eps,lst;

class AdvancedMazeKey10x10m42(AdvancedMazeKey):

    def __init__(self):
        super(AdvancedMazeKey10x10m42, self).__init__(rows=10,columns=10,wmode='mode42')

class AdvancedMazeKey9x9m42(AdvancedMazeKey):

    def __init__(self):
        super(AdvancedMazeKey9x9m42, self).__init__(rows=9,columns=9,wmode='mode42')

class AdvancedMazeKey8x8m42(AdvancedMazeKey):

    def __init__(self):
        super(AdvancedMazeKey8x8m42, self).__init__(rows=8,columns=8,wmode='mode42')

class AdvancedMazeKey7x7m42(AdvancedMazeKey):

    def __init__(self):
        super(AdvancedMazeKey7x7m42, self).__init__(rows=7,columns=7,wmode='mode42')

class AdvancedMazeKey6x6m42(AdvancedMazeKey):

    def __init__(self):
        super(AdvancedMazeKey6x6m42, self).__init__(rows=6,columns=6,wmode='mode42')

class AdvancedMazeKey5x5m42(AdvancedMazeKey):

    def __init__(self):
        super(AdvancedMazeKey5x5m42, self).__init__(rows=5,columns=5,wmode='mode42')

class AdvancedMazeKey4x4m42(AdvancedMazeKey):

    def __init__(self):
        super(AdvancedMazeKey4x4m42, self).__init__(rows=4,columns=4,wmode='mode42')



