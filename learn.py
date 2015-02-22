import parse as par
import mdpmodel as mod
import mdpsolve as sol


def learn(obs_, gamma=1, R=None):
    """Returns strategy and other data from observations

    takes 3d list of observations & reward list
    (if step-wise rewards not included in observations)
    [obs],[obs,gamma],[obs,gamma,R]"""
    if R is None:
        stateMap, actMap, obs = par.parse(obs_)
    else:
        stateMap, actMap, obs = par.parse(obs_, R)

    model = mod.model(len(stateMap), len(actMap), obs)

    P = model[0]
    R = model[1]
    policy = sol.policy(P, gamma, R)

    # map integer policy and action back to
    strat = {}
    for i in range(0, len(policy)):
        strat[stateMap[i]] = actMap[policy[i]]

    # return strategy, transition matrix, reward
    results = [strat, model[0], model[1], stateMap]
    return results
