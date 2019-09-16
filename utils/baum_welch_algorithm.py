import numpy as np
from .common import Common
from .forward_algorithm import ForwardAlgorithm
from .backward_algorithm import BackwardAlgorithm
#
#
#
#
#
# For learning the parameters A and B in the HMM model.
#
# Implements a version of the Estimation-Maximization (E-M) algorithm to learn.


class BaumWelchAlgorithm(Common):

    def __init__(self, N, T, A, B, V, O):
        super().__init__(N, T, A, B, O)
        self.V = V
        self.fa = ForwardAlgorithm(N, T, A, B, O)
        self.ba = BackwardAlgorithm(N, T, A, B, O)

    #
    # Uses Forward and Backward algorithms to calculate the probability that,
    # given the set of observations and the HMM model, the HMM is at some
    # state (i) at time (t) and transitions to another state (j) at time (t+1). γ
    #
    # γ_t(j) = αt(j)βt(j) / P(O|λ)

    def estimate_γ_at_t(self, t):
        γ = np.zeros((self.B.shape[0], self.T))

        γ[:, t] = self.fa.table[t, :] * self.ba.table[t, :] / \
            self.fa.table[self.T - 1][self.N - 1]

        return γ

    def estimate_ξ_at_t(self, t):

        # matrix that stores dragon result is same dimension as A
        ξ = np.zeros(self.A.shape)

        if t == self.T - 2:
            ξ[:, self.N-1] = self.fa.table[t, :] * \
                self.A[:, self.N-1] * self.ba.table[t][:] / \
                self.fa.table[self.T - 1][self.N - 1]
        else:
            for j in range(self.N):
                ξ[:, j] = self.fa.table[t, :] * \
                    self.A[:, j] * self.B[j][self.O[t]-1] * \
                    self.ba.table[t+1][j] / \
                    self.fa.table[self.T - 1][self.N - 1]
        return ξ

    def maximize(self):

        epsilon = 1 * 10**-8  # To prevent division by zero

        self.fa.forward()
        self.ba.backward()

        A_numerator = np.zeros(self.A.shape)
        A_denominator = np.zeros((self.A.shape[0], 1))

        B_numerator = np.zeros(self.B.shape)
        B_denominator = np.zeros((self.B.shape[0], 1))

        for t in range(self.T - 1):

            # A parameter
            A_denominator += np.sum(self.estimate_ξ_at_t(t),
                                    axis=1, keepdims=True)
            A_numerator += self.estimate_ξ_at_t(t)

            # B parameter
            B_denominator += np.sum(self.estimate_γ_at_t(t),
                                    axis=1, keepdims=True)
            for v in range(len(self.V)):
                if t > 0:
                    if self.O[t - 1] == self.V[v]:
                        B_numerator[:, v] += np.squeeze(np.sum(self.estimate_γ_at_t(t),
                                                               axis=1, keepdims=True))

        A_denominator += epsilon
        B_denominator += epsilon

        self.A = A_numerator / A_denominator
        self.B = B_numerator / B_denominator

        self.fa.update_parameters(self.A, self.B, self.O)
        self.ba.update_parameters(self.A, self.B, self.O)


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
    V = [1, 2, 3]
    # O = [1, 2, 3]

    O = [2, 3, 3, 2, 3, 2, 3, 2, 2, 3, 1, 3, 3, 1, 1, 1,
         2, 1, 1, 1, 3, 1, 2, 1, 1, 1, 2, 3, 3, 2, 3, 2, 2]

    bw = BaumWelchAlgorithm(A.shape[0], len(O), A, B, V, O)

    bw.pretty_print(bw.A_init, y_axis="state", table_name="A")
    bw.pretty_print(bw.B, y_axis="state",
                    x_axis="observation", table_name="B")

    bw.maximize()
    bw.pretty_print(bw.A, y_axis="state", table_name="A'")
    bw.pretty_print(bw.B, y_axis="state",
                    x_axis="observation", table_name="B'")

    # bw.maximize()
    # bw.pretty_print(bw.A, y_axis="state", table_name="A''")
    # bw.pretty_print(bw.B, y_axis="state",
    #                 x_axis="observation", table_name="B''")


if __name__ == "__main__":
    main()
