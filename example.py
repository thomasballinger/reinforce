import learn as l


def main():
    """
    >>> main()
    From these three paths, the learned strategy is: 
    {'A': 'L', 'C': 'F', 'B': 'L', 'Prize': 'F', 'D': 'L'}
    And the state-transition probability matrix is: 
    [[[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.2, 0.2, 0.2, 0.2, 0.2]], [[0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2]], [[1.0, 0.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2], [0.0, 0.0, 0.0, 1.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2]], [[0.0, 0.0, 0.0, 0.0, 1.0], [0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2]], [[0.2, 0.2, 0.2, 0.2, 0.2], [1.0, 0.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2], [0.0, 0.0, 0.0, 1.0, 0.0]]]
    """
    obs1 = [["A", "F", 0], ["A", "L", 0], ["Prize", "F", 1]]
    obs2 = [["C", "R", 0], ["D", "F", 0], ["B", "B", 0], ["D", "L", 0]]
    obs3 = [["C", "F", 0], ["A", "R", 0], ["B", "L", 0], ["A", "L", 0], ["Prize", "L", 1]]

    obs = [obs1, obs2, obs3]
    gamma = 0.95  # slight discount to rewards farther in the future

    model = l.learn(obs, gamma)
    # or try it without gamma
    # model = l.learn(obs)

    print ("From these three paths, the learned strategy is: ")
    print (model[0])

    print("And the state-transition probability matrix is: ")
    print(model[1])

    # note that many transition probabilities are estimated as uniform because there isn't yet data

if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.REPORT_UDIFF)
