#import logging
from gym.envs.registration import register

#logger = logging.getLogger(__name__)

register(
    id='1DMaze_s-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x16s',
    timestep_limit=2000,
)

register(
    id='1DMaze_c-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x16c',
    timestep_limit=2000,
)
