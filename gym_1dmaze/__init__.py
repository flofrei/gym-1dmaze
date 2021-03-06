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
    timestep_limit=100,
)

register(
    id='2DMaze_9x9_mode42-v0',
    entry_point='gym_1dmaze.envs:AdvancedMaze9x9m42',
    timestep_limit=100,
)

register(
    id='2DMaze_8x8_mode42-v0',
    entry_point='gym_1dmaze.envs:AdvancedMaze8x8m42',
    timestep_limit=100,
)

register(
    id='2DMaze_7x7_mode42-v0',
    entry_point='gym_1dmaze.envs:AdvancedMaze7x7m42',
    timestep_limit=100,
)

register(
    id='2DMaze_6x6_mode42-v0',
    entry_point='gym_1dmaze.envs:AdvancedMaze6x6m42',
    timestep_limit=100,
)

register(
    id='2DMaze_5x5_mode42-v0',
    entry_point='gym_1dmaze.envs:AdvancedMaze5x5m42',
    timestep_limit=100,
)

register(
    id='2DMaze_10x10_mode0-v0',
    entry_point='gym_1dmaze.envs:AdvancedMaze10x10m0',
    timestep_limit=100,
)

register(
    id='2DMaze_4x4_mode42-v0',
    entry_point='gym_1dmaze.envs:AdvancedMaze4x4m42',
    timestep_limit=100,
)

register(
    id='2DMaze_4x4_mode0-v0',
    entry_point='gym_1dmaze.envs:AdvancedMaze4x4m0',
    timestep_limit=100,
)

register(
    id='2DMaze_4x4_mode20-v0',
    entry_point='gym_1dmaze.envs:AdvancedMaze4x4m20',
    timestep_limit=100,
)

register(
    id='2DMaze_4x4_mode21-v0',
    entry_point='gym_1dmaze.envs:AdvancedMaze4x4m21',
    timestep_limit=100,
)

register(
    id='2DMaze_4x4_mode22-v0',
    entry_point='gym_1dmaze.envs:AdvancedMaze4x4m22',
    timestep_limit=100,
)

register(
    id='2DMaze_4x4_mode23-v0',
    entry_point='gym_1dmaze.envs:AdvancedMaze4x4m23',
    timestep_limit=100,
)

register(
    id='2DMaze_4x4_mode29-v0',
    entry_point='gym_1dmaze.envs:AdvancedMaze4x4m29',
    timestep_limit=100,
)

register(
    id='2DMaze_4x4_mode33-v0',
    entry_point='gym_1dmaze.envs:AdvancedMaze4x4m33',
    timestep_limit=100,
)

register(
    id='2DMaze_10x10_mode33-v0',
    entry_point='gym_1dmaze.envs:AdvancedMaze10x10m33',
    timestep_limit=100,
)

register(
    id='2DMazeLine_4x4_mode42-v0',
    entry_point='gym_1dmaze.envs:AdvancedMazeLine4x4m42',
    timestep_limit=100,
)

register(
    id='2DMazeLine_4x4_mode0-v0',
    entry_point='gym_1dmaze.envs:AdvancedMazeLine4x4m0',
    timestep_limit=100,
)

register(
    id='2DMazeLine_10x10_mode42-v0',
    entry_point='gym_1dmaze.envs:AdvancedMazeLine10x10m42',
    timestep_limit=100,
)

register(
    id='2DMazeLine_10x10_mode0-v0',
    entry_point='gym_1dmaze.envs:AdvancedMazeLine10x10m0',
    timestep_limit=100,
)

register(
    id='2DMazeKey_10x10_mode42-v0',
    entry_point='gym_1dmaze.envs:AdvancedMazeKey10x10m42',
    timestep_limit=100,
)

register(
    id='2DMazeKey_9x9_mode42-v0',
    entry_point='gym_1dmaze.envs:AdvancedMazeKey9x9m42',
    timestep_limit=100,
)

register(
    id='2DMazeKey_8x8_mode42-v0',
    entry_point='gym_1dmaze.envs:AdvancedMazeKey8x8m42',
    timestep_limit=100,
)

register(
    id='2DMazeKey_7x7_mode42-v0',
    entry_point='gym_1dmaze.envs:AdvancedMazeKey7x7m42',
    timestep_limit=100,
)

register(
    id='2DMazeKey_6x6_mode42-v0',
    entry_point='gym_1dmaze.envs:AdvancedMazeKey6x6m42',
    timestep_limit=100,
)

register(
    id='2DMazeKey_5x5_mode42-v0',
    entry_point='gym_1dmaze.envs:AdvancedMazeKey5x5m42',
    timestep_limit=100,
)

register(
    id='2DMazeKey_4x4_mode42-v0',
    entry_point='gym_1dmaze.envs:AdvancedMazeKey4x4m42',
    timestep_limit=100,
)
