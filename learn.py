import parse as par
import mdpmodel as mod
import mdpsolve as sol


def learn(obs, gamma=1, R=None):
    """Returns strategy and other data from observations

    takes 3d list of observations & reward list
    (if step-wise rewards not included in observations)
    [obs],[obs,gamma],[obs,gamma,R]

    obs has the structure
    [[[state, action, reward], [state, action, reward], ...],
     [[state, action, reward], [state, action, reward], ...], ...]
    """
    if R is not None:
        obs = par.obs_with_rewards(obs, R)
    stateMap, actMap, obs = par.number_states_and_actions(obs)

    P, R = mod.model(len(stateMap), len(actMap), obs)

    policy = sol.policy(P, gamma, R)

    # map integer policy and action back to
    strat = {}
    for i in range(0, len(policy)):
        strat[stateMap[i]] = actMap[policy[i]]

    # return strategy, transition matrix, reward
    results = [strat, P, R, stateMap]
    return results

if __name__ == '__main__':
    import example
    example.example2()
