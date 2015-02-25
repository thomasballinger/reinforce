from __future__ import division


def model(num_states, num_actions, obs):
    """Returns propability transition matrix and vector of rewards by state"""
    # initialize counters for reward and probability transition matrix
    s = num_states
    a = num_actions
    pcount = [[[0]*s for _ in range(a)] for _ in range(s)]
    rcount = [[0, 0] for _ in range(s)]

    # count state and reward observations
    for o in range(0, len(obs)):
        for t in range(0, len(obs[o])):
            # REWARD TRACKER:
            state, action, reward = obs[o][t]

            # increment cumulative reward for observed state
            rcount[state][0] += reward
            # increment state visits count
            rcount[state][1] += 1

            # PROBABILITY TRANSITION TRACKER
            # increment count of transitions, C_sa[s']
            if (t < (len(obs[o])-1)):
                state_ = obs[o][t+1][0]
                pcount[state][action][state_] += 1

    # compute R[s]
    R = [0]*s
    for i in range(0, s):
        R[i] = rcount[i][0]/rcount[i][1] if (rcount[i][1]) else (0)

    P = [[[0]*s for _ in range(a)] for _ in range(s)]
    # compute P_sa[s']
    for i in range(0, s):
        for j in range(0, a):
            visits = sum(pcount[i][j])
            for k in range(0, s):
                P[i][j][k] = pcount[i][j][k]/visits if (visits) else (1/s)

    return [P, R]
