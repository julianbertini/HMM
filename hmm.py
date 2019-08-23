import numpy as np
from baum_welch_algorithm import BaumWelchAlgorithm
from common import Common


class Hmm(Common):

    def __init__(self, A, B, V, O):
        """Initializes the HMM model with the required parameters and their
        initial conditions.

        Arguments:
            A {Numpy matrix} -- initial conditions for probabilities between the state transitions
            B {[type]} -- [description]
            V {[type]} -- [description]
            O {[type]} -- [description]
        """

        super().__init__(A.shape[0], len(O), A, B, O)
        self.model = BaumWelchAlgorithm(A.shape[0], len(O), A, B, V, O)

    def sync_parameters(self):
        self.A = self.model.A
        self.B = self.model.B

    def update_training_set(self, O):
        self.model.O = O

    def maximize(self):
        self.model.maximize()
        self.sync_parameters()


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

    hmm = Hmm(A, B, V, O)

    hmm.maximize()

    hmm.pretty_print(hmm.A,  y_axis="state", table_name="A'")
    hmm.pretty_print(hmm.B,  y_axis="state",
                     x_axis="observation", table_name="B'")

    hmm.maximize()


if __name__ == "__main__":
    main()
