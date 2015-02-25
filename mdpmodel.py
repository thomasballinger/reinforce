from __future__ import division
from collections import Counter


def model(num_states, num_actions, obs):
    """Returns propability transition matrix and vector of rewards by state"""
    # initialize counters for reward and probability transition matrix
    pcount = [[[0]*num_states
               for _ in range(num_actions)]
              for _ in range(num_states)]

    state_visits = Counter(state for ob in obs for state, _, _ in ob)
    state_rewards = Counter()
    for ob in obs:
        for state, _, reward in ob:
            state_rewards.update({state: reward})

    # count state and reward observations
    for ob in obs:
        for (state, action, _), (next_state, _, _) in zip(ob[:-1], ob[1:]):
            pcount[state][action][next_state] += 1

    # compute R[s]
    R = [state_rewards[i]/state_visits[i] if state_visits[i] else 0
         for i in range(num_states)]

    # P[initial_state][action][dest_state] = probability
    P = [[[0]*num_states for _ in range(num_actions)] for _ in range(num_states)]
    # compute P_sa[s']
    for initial_state_p, initial_state_count in zip(P, pcount):
        for action_p, action_count in zip(initial_state_p, initial_state_count):
            visits = sum(action_count)
            for k in range(0, num_states):
                action_p[k] = action_count[k] / visits if visits else 1 / num_states

    return [P, R]
