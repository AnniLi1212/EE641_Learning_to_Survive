from gymnasium.envs.registration import register
from .game import SurvivalGameEnv
register(
    id='SurvivalGame-v0',
    entry_point='src.environment.game:SurvivalGameEnv',
    max_episode_steps=1000,
)
