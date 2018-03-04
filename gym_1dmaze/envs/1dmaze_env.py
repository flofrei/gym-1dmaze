"""
World Setup Toy Example

ASCII based features and impl.
"""

import numpy as np

from gym import core, spaces
from gym.envs.registration import register

class 1DMaze(object):
    def __init__(self,wsize=16,expansion_flag=False):
        super(SimpleWorld, self).__init__()

        if expansion_flag==True:
            self.action_space = ['left', 'right', 'leftskip', 'rightskip'];
        else:
            self.action_space = ['left', 'right'];
        
        self.n_actions = len(self.action_space);
        #self.title('1D_world')
        self.agent_position=-1;
        self.goal_position=-1;
        self.world_size=wsize;
        self.world=self.create_world();
        self.set_startposition(3);
        self.set_goalposition(15);

    def create_world(self):
        listholder = [];
        for k in range(self.world_size):
            listholder+=['_'];
            
        #env_list = ['_']*size;
        return listholder;

    def swap(self,pos1,pos2):
            self.world[pos1],self.world[pos2] = self.world[pos2],self.world[pos1];

    def set_startposition(self,position):
        self.world[position] = 'a';
        self.agent_position=position;

    def set_goalposition(self,position):
        self.world[position] = 'T';
        self.goal_position=position;

    def reset(self):
        self.world=self.create_world();

    def new_print(self):
        #size = len(world);
        newstring = ''.join(self.world);
        print(newstring);

    def get_features(self):
        new_input = np.zeros((1,self.world_size)); 
        for r in range(self.world_size):
            new_input[0,r] = ord(self.world[r]);

        return new_input 


    def step(self,action):
        #Step function in the 1D World

        ## First action value
        if action == 'left':
            ## Going out of the world case
            if self.agent_position==0:
                r = 0; 
                return self.get_features(), r;
            ## Moving internally in the world
            else:
                ## Moving to goal position 
                if self.agent_position==self.goal_position+1: 
                    r = 1;
                    return 'terminal', r;
                ## only moving around  
                else:
                    r = 0;
                    self.swap(self.agent_position,self.agent_position-1);
                    self.agent_position=self.agent_position-1;
                    return self.get_features(), r;
        ## Second action value        
        elif action == 'right':
            ## Going out of the world case
            if self.agent_position==self.world_size-1:
                r = 0;
                return self.get_features(),r;
            ## Moving internally in the World
            else:
                ## Moving to goal position
                if self.agent_position==self.goal_position-1:
                    r = 1;
                    return 'terminal', r;
                ## only moving around
                else:
                    r = 0;
                    self.swap(self.agent_position,self.agent_position+1);
                    self.agent_position=self.agent_position+1;
                    return self.get_features(), r;
        ## Third action value  
        elif action == 'leftskip':
            ## Goind out of the world
            if self.agent_position==1 or self.agent_position==0:
                r = 0;
                return self.get_features(), r;
            ## Moving internally in the world
            else:
                ## Moving to goal position
                if self.agent_position == self.goal_position+2:
                    r = 1;
                    return 'terminal', r;
                ## only moving around
                else:
                    r = 0;
                    self.swap(self.agent_position,self.agent_position-2);
                    self.agent_position=self.agent_position-2;
                    return self.get_features(), r;
        ## Fourth action value 
        elif action == 'rightskip':
            ## Goind out of the world
            if self.agent_position==self.world_size-1 or self.agent_position==self.world_size-2:
                r = 0;
                return self.get_features(), r;
            ## Moving internally in the world
            else:
                ## Moving to goal position
                if self.agent_position == self.goal_position-2:
                    r = 1;
                    return 'terminal', r;
                ## only moving around
                else:
                    r = 0;
                    self.swap(self.agent_position,self.agent_position+2);
                    self.agent_position=self.agent_position+2;
                    return self.get_features(), r;


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


#register(
#    id='1Dsimpleworld-v0',
#    entry_point='world.simpleworld:SimpleWorld',
#   timestep_limit=20000,
#   reward_threshold=1,
#)
#if __name__ == "__main__":
    #size=32;
#    env = SimpleWorld();
    #uncomment for analysis
    #test_suite();
    
