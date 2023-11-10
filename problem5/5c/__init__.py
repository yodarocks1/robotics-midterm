"""
Usage: python3 -m 5c [preset] [--render|--preset|--out|--episodes]

Positional Arguments:
  preset (default: CartPole-v1)
    This flag sets what environment should be created and trained against.
    For the sake of this question, the relevant presets are:
      CartPole-v1
      CartPoleMod-v1
      FrozenLake-v1
      FrozenLakeMod-v1

Flags:
  --render [mode] (default: None)
      Including this flag without specifying a mode will set a render mode of "human".
  --preset <id> (default: CartPole-v1)
      Alias for the positional argument preset
  --out <path> (default: model.pth)
      Where the model, once trained, should be saved
  --episodes <n> (default: 1000)
      The number of episodes for which to train the model
  --<ARG> [VALUE] [VALUE...]
      Sets the keyword argument when creating the environment
      If no values are provided (--<ARG>):
        ARG=True
      If one value is provided (--<ARG> <VALUE>):
        ARG=VALUE
      If more values are provided (--<ARG> <VALUE> <VALUE> <...>):
        ARG=[VALUE, VALUE, ...]

Environment-specific
  CartPoleMod-v1:
    --left_multiplier (default: 1)
        Multiplier for the force when going to the left
    --right_multiplier (default: 2)
        Multiplier for the force when going to the right

  FrozenLake-v1:
    --map_name [map_name]
        Use a preloaded map
        If not provided, generates a random 8x8 map with 80% of locations frozen
    --is_slippery
        Sets the environment variable "is_slippery" to True
    --not_is_slippery
        Sets the environment variable "is_slippery" to False

  FrozenLakeMod-v1:
    --map_name [map_name]
        Use a preloaded map
        If not provided, generates a random 8x8 map with 80% of locations frozen
    --is_slippery
        Sets the environment variable "is_slippery" to True
    --not_is_slippery
        Sets the environment variable "is_slippery" to False
    --death_probability <probability> (default: 0.02)
        The probability that the agent will randomly die of cold
"""

from gymnasium.envs.registration import register

register(
    id="CartPoleMod-v1",
    entry_point="5c.envs:CartPoleEnvModified",
    max_episode_steps=500,
    reward_threshold=475.0,
    kwargs={"right_multiplier": 2.}
)
register(
    id="FrozenLakeMod-v1",
    entry_point="5c.envs:FrozenLakeEnvModified",
    max_episode_steps=100,
    reward_threshold=0.70,
    kwargs={"map_name": "4x4"},
)

