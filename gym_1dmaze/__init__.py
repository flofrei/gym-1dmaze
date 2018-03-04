import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='1DMaze-v0',
    entry_point='gym_1dmaze.envs:1DMaze',
    timestep_limit=1000,
    reward_threshold=1.0,
    nondeterministic = True,
)
