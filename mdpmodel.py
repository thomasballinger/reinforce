from __future__ import division


def model(num_states, num_actions, obs):
    """Returns propability transition matrix and vector of rewards by state"""
    # initialize counters for reward and probability transition matrix
    pcount = [[[0]*num_states
               for _ in range(num_actions)]
              for _ in range(num_states)]
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
    R = [0]*num_states
    for i in range(num_states):
        R[i] = rcount[i][0]/rcount[i][1] if (rcount[i][1]) else (0)

    P = [[[0]*num_states for _ in range(num_actions)] for _ in range(num_states)]
    # compute P_sa[s']
    for i in range(0, num_states):
        for j in range(0, num_actions):
            visits = sum(pcount[i][j])
            for k in range(0, num_states):
                P[i][j][k] = pcount[i][j][k]/visits if (visits) else (1/num_states)

    return [P, R]
