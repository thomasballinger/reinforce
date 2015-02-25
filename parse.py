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


def number_states_and_actions(obs):
    """Augment 3D obs array with states and actions

    Returns state map and actions map, and adds states and actions to obs"""

    stateMap = get_state_map(obs)
    state_num = {s: i for i, s in enumerate(stateMap)}
    actionMap = get_action_map(obs)
    action_num = {a: i for i, a in enumerate(actionMap)}

    obs = [[[state_num[state], action_num[action], reward]
            for state, action, reward in ob]
           for ob in obs]

    return stateMap, actionMap, obs


def get_action_map(obs):
    """Adds actions to observations and return action map"""
    return uniq(step[1] for ob in obs for step in ob)


def get_state_map(obs):
    """Adds states to observations and return state map"""
    return uniq(step[0] for ob in obs for step in ob)


def uniq(seq):
    "Returns unique elements preserving order"
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
