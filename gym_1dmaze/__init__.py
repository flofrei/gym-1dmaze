#import logging
from gym.envs.registration import register

#logger = logging.getLogger(__name__)

register(
    id='1DMaze_1x16_simpleactionset_mode0-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x16sasm0',
    timestep_limit=100,
)

register(
    id='1DMaze_1x16_extendedactionset_mode0-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x16easm0',
    timestep_limit=100,
)

register(
    id='1DMaze_1x32_simpleactionset_mode0-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x32sasm0',
    timestep_limit=100,
)

register(
    id='1DMaze_1x32_extendedactionset_mode0-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x32easm0',
    timestep_limit=100,
)

register(
    id='1DMaze_1x16_simpleactionset_mode10-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x16sasm10',
    timestep_limit=100,
)

register(
    id='1DMaze_1x16_extendedactionset_mode10-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x16easm10',
    timestep_limit=100,
)

register(
    id='1DMaze_1x32_simpleactionset_mode10-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x32sasm10',
    timestep_limit=100,
)

register(
    id='1DMaze_1x32_extendedactionset_mode10-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x32easm10',
    timestep_limit=100,
)

register(
    id='1DMaze_1x16_simpleactionset_mode42-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x16sasm42',
    timestep_limit=100,
)

register(
    id='1DMaze_1x16_extendedactionset_mode42-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x16easm42',
    timestep_limit=100,
)

register(
    id='1DMaze_1x32_simpleactionset_mode42-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x32sasm42',
    timestep_limit=100,
)

register(
    id='1DMaze_1x32_extendedactionset_mode42-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x32easm42',
    timestep_limit=100,
)

register(
    id='1DMaze_1x4_simpleactionset_mode42-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x4sasm42',
    timestep_limit=100,
)

register(
    id='1DMaze_1x4_extendedactionset_mode42-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x4easm42',
    timestep_limit=100,
)

register(
    id='1DMaze_1x4_simpleactionset_mode0-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x4sasm0',
    timestep_limit=100,
)

register(
    id='1DMaze_1x4_extendedactionset_mode0-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x4easm0',
    timestep_limit=100,
)

register(
    id='1DMaze_1x10_simpleactionset_mode42-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x10sasm42',
    timestep_limit=100,
)

register(
    id='1DMaze_1x10_simpleactionset_mode0-v0',
    entry_point='gym_1dmaze.envs:SimpleMaze1x10sasm0',
    timestep_limit=100,
)

register(
    id='2DMaze_10x10_mode42-v0',
    entry_point='gym_1dmaze.envs:AdvancedMaze10x10m42',
    timestep_limit=50,
)

register(
    id='2DMaze_10x10_mode0-v0',
    entry_point='gym_1dmaze.envs:AdvancedMaze10x10m0',
    timestep_limit=50,
)
