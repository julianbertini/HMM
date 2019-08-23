import numpy as np
from common import Common
#
#
#
#
# NOTE: first and last states are START and STOP states.


class ForwardAlgorithm(Common):

    def __init__(self, N, T, A, B, O):
        super().__init__(N, T, A, B, O)
        self.table = self.initialize_dynamic_table()

    def forward(self):

        # Fill up dynamic table
        for t in range(2, self.T-1):
            for j in range(1, self.N):
                self.table[t][j] = (np.dot(
                    self.table[t - 1, :], self.A[:, j]) *
                    self.B[j][self.O[t-1]-1])

        self.table[self.T-1][self.N-1] = np.dot(
            self.table[self.T - 2, :], self.A[:, self.N-1])


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

    fa = ForwardAlgorithm(A.shape[0], len(O), A, B, O)
    fa.forward()

    fa.pretty_print(fa.table)


if __name__ == "__main__":
    main()
