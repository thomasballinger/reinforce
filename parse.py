from __future__ import division
"""
create maps from states<->ints, actions<->ints
convert single reward to step rewards if applicable
"""


def add_states_and_actions(obs, R=None):
    """Augment 3D obs array with states and actions

    Returns state map and actions map, and adds states and actions to obs"""
    if R is not None:
        add_rewards_to_obs(obs, R)
    stateMap = add_states_to_obs(obs)
    actMap = add_actions_to_obs(obs)
    return [stateMap, actMap]


def add_rewards_to_obs(obs, R):
    """Adds reward values to R

    obs and R, 1 reward per observation
    convert obs to include rewards each step
    """
    for o in range(0, len(obs)):
        totalReward = R[o]
        stepReward = totalReward/len(obs[o])
        for t in range(0, len(obs[o])):
            obs[o][t].append(stepReward)


def add_actions_to_obs(obs_with_rewards):
    """Adds actions to observations and return action map"""
    # create maps from actions and states to integers
    obs = obs_with_rewards
    actMap = []
    for o in range(0, len(obs)):
        for t in range(0, len(obs[o])):
            action = obs[o][t][1]
            if (action not in actMap):
                actMap.append(action)
            obs[o][t][1] = actMap.index(action)
    return actMap


def add_states_to_obs(obs_with_rewards):
    """Adds states to observations and return state map"""
    obs = obs_with_rewards
    stateMap = []
    for o in range(0, len(obs)):
        for t in range(0, len(obs[o])):
            state = obs[o][t][0]
            if (state not in stateMap):
                stateMap.append(state)
            obs[o][t][0] = stateMap.index(state)
    return stateMap
