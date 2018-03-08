#import logging
from gym.envs.registration import register

#logger = logging.getLogger(__name__)

register(
    id='1DMaze_1x16_simpleactionset_mode0-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x16sasm0',
    timestep_limit=2000,
)

register(
    id='1DMaze_1x16_extendedactionset_mode0-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x16easm0',
    timestep_limit=2000,
)

register(
    id='1DMaze_1x32_simpleactionset_mode0-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x32sasm0',
    timestep_limit=2000,
)

register(
    id='1DMaze_1x32_extendedactionset_mode0-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x32easm0',
    timestep_limit=2000,
)
