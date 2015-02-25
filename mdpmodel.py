from __future__ import division


def model(num_states, num_actions, obs):
    """Returns propability transition matrix and vector of rewards by state"""
    # initialize counters for reward and probability transition matrix
    pcount = [[[0]*num_states
               for _ in range(num_actions)]
              for _ in range(num_states)]
    # [total_reward, state_visit_count
    rcount = [[0, 0] for _ in range(num_states)]

    # count state and reward observations
    for ob in obs:
        for step in ob:
            # REWARD TRACKER:
            state, action, reward = step

            # increment cumulative reward for observed state
            rcount[state][0] += reward
            # increment state visits count
            rcount[state][1] += 1

            # PROBABILITY TRANSITION TRACKER
            # increment count of transitions, C_sa[s']
        for (state, action, _), (next_state, _, _) in zip(ob[:-1], ob[1:]):
            pcount[state][action][next_state] += 1

    # compute R[s]
    R = [rcount[i][0]/rcount[i][1] if (rcount[i][1]) else (0)
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
