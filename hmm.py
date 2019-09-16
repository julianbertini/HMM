import numpy as np
from utils.baum_welch_algorithm import BaumWelchAlgorithm
from utils.common import Common


class Hmm(Common):

    def __init__(self, A, B, V, O):
        """ Initializes the HMM model with the required parameters and their
        initial conditions.

        Arguments:
            A {Numpy matrix} -- initial conditions for probabilities 
                                between the state transitions
            B {Numpy matrix} -- initial conditions for probabilities of observations
            V {Array} -- vector defining the kinds of observations allowed 
                            (so defines the observation space)
            O {Array} -- observation vector, indicating the sequence of 
                            observations to train the model with
        """

        super().__init__(A.shape[0], len(O), A, B, O)
        self.model = BaumWelchAlgorithm(A.shape[0], len(O), A, B, V, O)

    def sync_parameters(self):
        """ Updates the model's parameters after each optimization
        """

        self.A = self.model.A
        self.B = self.model.B

    def update_training_set(self, O):
        """ Updates the training set for the model

        Arguments:
            O {Array} -- the new array of training observations
        """

        self.model.O = O

    def maximize(self):
        """ Executes the E-M algorithm to update the A and B 
            probability matrices
        """

        self.model.maximize()
        self.sync_parameters()


def main():
    """ Here is a sample use of the model to train initial probability
        matrices {A} and {B} on a series of observations {O}. We define the 
        observation space in {V}, meaning that, in this example, [1,2,3]
        are the "allowed" or "defined" observations that the system can work with.
    """

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

    # We initialize the model here
    hmm = Hmm(A, B, V, O)

    # Call maximize to train the model (this is 1 iteration of training)
    hmm.maximize()

    # Use the pretty_print function to print the probability matrices
    # after the round of training
    hmm.pretty_print(hmm.A,  y_axis="state", table_name="A'")
    hmm.pretty_print(hmm.B,  y_axis="state",
                     x_axis="observation", table_name="B'")

    # We can call maximize() again to do another iteration of training.
    # This could be done in a loop, and we could switch up the trianing
    # observation set with update_training_set()
    hmm.maximize()


if __name__ == "__main__":
    main()
