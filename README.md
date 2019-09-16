# HMM
A prototype implementation of a Hidden Markov Model

# Some Notes
- I've tested the model with results shown by Eisner in his paper on Markov models. It can be found [here](https://cs.jhu.edu/~jason/papers/).
- I've added some comments to the `hmm.py` with information about how to use the model. This is very barebones documentation, and I'll add a more extensive how-to guide on this readme soon. 
# Some Explanation on the Example in `main` of `hmm.py`
- The example implemented in the `main` function of the `hmm.py` file uses the canonical example of modeling the number of ice creams that are eaten on a given day based upon whether the day was hot or cold. As such, `V = [1,2,3]` indicates that the observation space is: either 1, 2, or 3 ice creams were eaten. We may observe `O = [1,3,3,2,3,1,2,3]`, which indicates that on the first day, 1 ice cream was eaten, then 3 ice creams on the second day, and so on. 
- The transition matrix `A` indicates that the probability of going from one state to another: in this case, it indicates the probability of going from a cold day to a hot day, or a hot day to a cold day, etc. 
- The matrix `B` indicates the probability of observing any given observation (in this case, number of ice creams eaten) from any given day. For example, it might indicate that the probability of observing 3 ice creams eaten on a cold day is low, at 10% percent.
- Although the example shown here is about ice creams, this model may be used to do `pattern of life` analysis, for instead o days, each state could indicate the particular state of a computer, and the observations may be how the computer got to that state. For example, the computer may be in a "logged-in" state, where the observation indicates that it was logged into remotely. 

# Current Limitations and Room for Improvement
- The current model requires one to preemtively define the number of states of the system. This may not be desirable if all the possible states of the system are not known a-priori. There is a way to create this Markov model such that it not only learns the probabilities (which is what this does now), but also learns the states in an unsupervised way. 
-  I think the best way to initialize the probabilities is just randomly, but there may be better ways. 
- There are many more variations that this model can take. See the overview of the original paper on Markov models below for more information on the different variations of this model. 

## Sources
- [See "An interactive spreadsheet for teaching the forward-backward algorithm"](https://cs.jhu.edu/~jason/papers/eisner.tnlp02.pdf)
- [Nice chapter on the theory behind HMM](https://cs.jhu.edu/~jason/papers/jurafsky+martin.slp3draft.ch9.pdf)
- [Another paper that presents the theory in a different way](http://cs229.stanford.edu/section/cs229-hmm.pdf)
-[Overview of theory of original paper](https://ieeexplore.ieee.org/abstract/document/18626)
