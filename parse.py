from __future__ import division
"""
create maps from states<->ints, actions<->ints
convert single reward to step rewards if applicable
"""


def obs_with_rewards(obs, R):
    """Returns observations with rewards embedded

    obs and R, 1 reward per observation
    convert obs to include rewards each step
    """
    if len(obs[0][0]) != 2:
        raise ValueError("obs has wrong dimensions: %d", obs[0][0])

    def stepReward(total, num_steps):
        return total / num_steps

    return [[step + [stepReward(totalReward, len(ob))] for step in ob]
            for ob, totalReward in zip(obs, R)]


def add_states_and_actions(obs, R=None):
    """Augment 3D obs array with states and actions

    Returns state map and actions map, and adds states and actions to obs"""
    if R is not None:
        obs = obs_with_rewards(obs, R)
    stateMap = add_states_to_obs(obs)
    actMap = add_actions_to_obs(obs)
    return [stateMap, actMap]


def add_actions_to_obs(obs_with_rewards):
    """Adds actions to observations and return action map"""
    # create maps from actions and states to integers
    obs = obs_with_rewards
    actMap = []
    for ob in obs:
        for step in ob:
            action = step[1]
            if (action not in actMap):
                actMap.append(action)
            step[1] = actMap.index(action)
    return actMap


def add_states_to_obs(obs_with_rewards):
    """Adds states to observations and return state map"""
    obs = obs_with_rewards
    stateMap = []
    for ob in obs:
        for step in ob:
            state = step[0]
            if (state not in stateMap):
                stateMap.append(state)
            step[0] = stateMap.index(state)
    return stateMap
