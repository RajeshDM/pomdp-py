from pomdp_py.framework.basics import GenerativeDistribution
import sys
import random as rand
import numpy as np

class Histogram(GenerativeDistribution):

    """
    Histogram representation of a probability distribution.

    __init__(self, histogram)

    Args:
        histogram (dict) is a dictionary mapping from
            variable value to probability
    """

    def __init__(self, histogram):
        """`histogram` is a dictionary mapping from
        variable value to probability"""
        if not (isinstance(histogram, dict)):
            raise ValueError("Unsupported histogram representation! %s"
                             % str(type(histogram)))
        self._histogram = histogram

    @property
    def histogram(self):
        """histogram(self)"""
        return self._histogram

    def __str__(self):
        return str(self._histogram)

    def __len__(self):
        return len(self._histogram)

    def __getitem__(self, value):
        """__getitem__(self, value)
        Returns the probability of `value`."""
        if value in self._histogram:
            return self._histogram[value]
        else:
            return 0

    def __setitem__(self, value, prob):
        """__setitem__(self, value, prob)
        Sets probability of value to `prob`."""
        self._histogram[value] = prob

    def __eq__(self, other):
        if not isinstance(other, Histogram):
            return False
        else:
            return self.histogram == other.histogram

    def __iter__(self):
        return iter(self._histogram)

    def mpe(self):
        """mpe(self)
        Returns the most likely value of the variable.
        """
        return max(self._histogram, key=self._histogram.get)

    def random(self, rnd=rand):
        """
        random(self)
        Randomly sample a value based on the probability
        in the histogram"""
        candidates = list(self._histogram.keys())
        prob_dist = []
        for value in candidates:
            prob_dist.append(self._histogram[value])

        np.random.seed(10)
        #if rnd == rand:
        rnd.seed(10)
        if sys.version_info[0] >= 3 and sys.version_info[1] >= 6:
            # available in Python 3.6+
            return rnd.choices(candidates, weights=prob_dist, k=1)[0]
        else:
            print("Warning: Histogram.random() using np.random"\
                  "due to old Python version (<3.6). Random seed ignored.")
            return np.random.choice(candidates, 1, p=prob_dist)[0]

    def get_histogram(self):
        """get_histogram(self)
        Returns a dictionary from value to probability of the histogram"""
        return self._histogram

    # Deprecated; it's assuming non-log probabilities
    def is_normalized(self, epsilon=1e-9):
        """Returns true if this distribution is normalized"""
        prob_sum = sum(self._histogram[state] for state in self._histogram)
        return abs(1.0-prob_sum) < epsilon




def abstraction_over_histogram(current_histogram, state_mapper):
    state_mappings = {s:state_mapper(s) for s in current_histogram}
    hist = {}
    for s in current_histogram:
        a_s = state_mapper(s)
        if a_s not in hist[a_s]:
            hist[a_s] = 0
        hist[a_s] += current_histogram[s]
    return hist

def update_histogram_belief(current_histogram, 
                            real_action, real_observation,
                            observation_model, transition_model, oargs={},
                            targs={}, normalize=True, static_transition=False,
                            next_state_space=None):
    """
    update_histogram_belief(current_histogram, real_action, real_observation,
                            observation_model, transition_model, oargs={},
                            targs={}, normalize=True, deterministic=False)
    This update is based on the equation:
    :math:`B_{new}(s') = n O(z|s',a) \sum_s T(s'|s,a)B(s)`.

    Args:
        current_histogram (~pomdp_py.representations.distribution.Histogram)
            is the Histogram that represents current belief.
        real_action (~pomdp_py.framework.basics.Action)
        real_observation (~pomdp_py.framework.basics.Observation)
        observation_model (~pomdp_py.framework.basics.ObservationModel)
        transition_model (~pomdp_py.framework.basics.TransitionModel)
        oargs (dict) Additional parameters for observation_model (default {})
        targs (dict) Additional parameters for transition_model (default {})
        normalize (bool) True if the updated belief should be normalized
        static_transition (bool) True if the transition_model is treated as static;
            This basically means Pr(s'|s,a) = Indicator(s' == s). This then means
            that sum_s Pr(s'|s,a)*B(s) = B(s'), since s' and s have the same state space.
            This thus helps reduce the computation cost by avoiding the nested iteration
            over the state space; But still, updating histogram belief requires
            iteration of the state space, which may already be prohibitive.
        next_state_space (set) the state space of the updated belief. By default,
            this parameter is None and the state space given by current_histogram
            will be directly considered as the state space of the updated belief.
            This is useful for space and time efficiency in problems where the state
            space contains parts that the agent knows will deterministically update,
            and thus not keeping track of the belief over these states.

    Returns:
        Histogram: the histogram distribution as a result of the update
    """
    new_histogram = {}  # state space still the same.
    total_prob = 0
    if next_state_space is None:
        next_state_space = current_histogram
    for next_state in next_state_space:
        observation_prob = observation_model.probability(real_observation,
                                                         next_state,
                                                         real_action,
                                                         **oargs)
        if not static_transition:
            transition_prob = 0
            for state in current_histogram:
                transition_prob += transition_model.probability(next_state,
                                                                state,
                                                                real_action,
                                                                **targs) * current_histogram[state]
        else:
            transition_prob = current_histogram[next_state]
            
        new_histogram[next_state] = observation_prob * transition_prob
        total_prob += new_histogram[next_state]

    # Normalize
    if normalize:
        for state in new_histogram:
            if total_prob > 0:
                new_histogram[state] /= total_prob
    return Histogram(new_histogram)
