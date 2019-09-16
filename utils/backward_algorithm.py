import numpy as np
from .common import Common
from .forward_algorithm import ForwardAlgorithm
#
#
#
#
# NOTE: first and last states are START and STOP states.


class BackwardAlgorithm(Common):

    def __init__(self, N, T, A, B, O):
        super().__init__(N, T, A, B, O)
        self.table = self.initialize_dynamic_table(
            init_type="backward")

    def backward(self):

        # Fill up dynamic table
        for t in reversed(range(1, self.T - 2)):
            for i in range(1, self.N):
                prob_sum = 0
                for j in range(1, self.N):
                    prob_sum = (prob_sum + self.table[t + 1, j] *
                                self.A[i, j] * self.B[j, self.O[t] - 1])
                self.table[t][i] = prob_sum

        prob_sum = 0
        for j in range(1, self.N):
            prob_sum = (prob_sum + self.table[1, j] *
                        self.A[0, j] * self.B[j, self.O[0] - 1])
        self.table[0][0] = prob_sum


def main():

    A = np.array([[0.0, 0.5, 0.5, 0.0],
                  [0.0, 0.8, 0.1, 0.1],
                  [0.0, 0.1, 0.8, 0.1],
                  [0.0, 0.0, 0.0, 0.0]])

    B = np.array([[0.0, 0.0, 0.0],
                  [0.7, 0.2, 0.1],
                  [0.1, 0.2, 0.7],
                  [0.0, 0.0, 0.0]])

    # only probabilities to whatever these observations are matter
    O = [1, 2, 3]

    ba = BackwardAlgorithm(A.shape[0], len(O), A, B, O)
    ba.backward()

    ba.pretty_print(ba.table)


if __name__ == "__main__":
    main()
