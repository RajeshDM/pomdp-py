
#from pomdp_py.framework.basics import Action, Agent, POMDP, State, Observation,\
#    ObservationModel, TransitionModel, GenerativeDistribution, PolicyModel,\
#    sample_generative_model
from pomdp_py.framework.basics import Action, Agent, POMDP, State, Observation,\
    ObservationModel, TransitionModel, GenerativeDistribution, PolicyModel
from pomdp_problems.rearrange_pomdp.algorithm.debugging import TreeDebugger
    
from pomdp_py.framework.planner import Planner
#from pomdp_py.representations.distribution.particles import Particles
from pomdp_problems.rearrange_pomdp.domain.action import *
import time
import random
import math
from tqdm import tqdm
from icecream import ic
from pomdp_problems.rearrange_pomdp.algorithm.nodes import TreeNode,QNode,VNode,RootVNode


class ActionPrior:
    """A problem-specific object"""

    def get_preferred_actions(self, state, history):
        """
        get_preferred_actions(cls, state, history, kwargs)
        Intended as a classmethod.
        This is to mimic the behavior of Simulator.Prior
        and GenerateLegal/GeneratePreferred in David Silver's
        POMCP code.

        Returns a set of tuples, in the form of (action, num_visits_init, value_init)
        that represent the preferred actions.
        In POMCP, this acts as a history-based prior policy,
        and in DESPOT, this acts as a belief-based prior policy.
        For example, given certain state or history, only it
        is possible that only a subset of all actions is legal;
        This is useful when there is domain knowledge that can
        be used as a heuristic for planning. """
        raise NotImplementedError


class RolloutPolicy(PolicyModel):
    def rollout(self, state, history):
        """rollout(self, State state, tuple history=None)"""
        pass

class RandomRollout(RolloutPolicy):
    """A rollout policy that chooses actions uniformly at random from the set of
    possible actions."""
    def rollout(self,state,history):
        """rollout(self, State state, tuple history=None)"""
        random.seed(15)
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]

class POUCT(Planner):

    """ POUCT (Partially Observable UCT) :cite:`silver2010monte` is presented in the POMCP
    paper as an extension of the UCT algorithm to partially-observable domains
    that combines MCTS and UCB1 for action selection.

    POUCT only works for problems with action space that can be enumerated.

    __init__(self,
             max_depth=5, planning_time=1., num_sims=-1,
             discount_factor=0.9, exploration_const=math.sqrt(2),
             num_visits_init=1, value_init=0,
             rollout_policy=RandomRollout(),
             action_prior=None, show_progress=False, pbar_update_interval=5)

    Args:
        max_depth (int): Depth of the MCTS tree. Default: 5.
        planning_time (float), amount of time given to each planning step (seconds). Default: -1.
            if negative, then planning terminates when number of simulations `num_sims` reached.
            If both `num_sims` and `planning_time` are negative, then the planner will run for 1 second.
        num_sims (int): Number of simulations for each planning step. If negative,
            then will terminate when planning_time is reached.
            If both `num_sims` and `planning_time` are negative, then the planner will run for 1 second.
        rollout_policy (RolloutPolicy): rollout policy. Default: RandomRollout.
        action_prior (ActionPrior): a prior over preferred actions given state and history.
        show_progress (bool): True if print a progress bar for simulations.
        pbar_update_interval (int): The number of simulations to run after each update of the progress bar,
            Only useful if show_progress is True; You can set this parameter even if your stopping criteria
            is time.
    """

    def __init__(self,
                 max_depth=5, planning_time=-1., num_sims=-1,
                 discount_factor=0.9, exploration_const=math.sqrt(2),
                 num_visits_init=0, value_init=0,
                 rollout_policy=RandomRollout(),
                 action_prior=None, show_progress=False, pbar_update_interval=5):
        self._max_depth = max_depth
        self._planning_time = planning_time
        self._num_sims = num_sims
        if self._num_sims < 0 and self._planning_time < 0:
            self._planning_time = 1.
        self._num_visits_init = num_visits_init
        self._value_init = value_init
        self._rollout_policy = rollout_policy
        self._discount_factor = discount_factor
        self._exploration_const = exploration_const
        self._action_prior = action_prior

        self._show_progress = show_progress
        self._pbar_update_interval = pbar_update_interval

        # to simplify function calls; plan only for one agent at a time
        self._agent = None
        self._last_num_sims = -1
        self._last_planning_time = -1

    @property
    def updates_agent_belief(self):
        return False

    @property
    def last_num_sims(self):
        """Returns the number of simulations ran for the last `plan` call."""
        return self._last_num_sims

    @property
    def last_planning_time(self):
        """Returns the amount of time (seconds) ran for the last `plan` call."""
        return self._last_planning_time

    def plan(self, agent):

        self._agent = agent   # switch focus on planning for the given agent
        if not hasattr(self._agent, "tree"):
            self._agent.add_attr("tree", None)
        action, time_taken, sims_count = self._search()
        self._last_num_sims = sims_count
        self._last_planning_time = time_taken
        return action

    def update(self, agent, real_action, real_observation):
        """
        update(self, Agent agent, Action real_action, Observation real_observation)
        Assume that the agent's history has been updated after taking real_action
        and receiving real_observation.
        """
        if not hasattr(agent, "tree") or agent.tree is None:
            print("Warning: agent does not have tree. Have you planned yet?")
            return

        if real_action not in agent.tree\
           or real_observation not in agent.tree[real_action]:
            agent.tree = None  # replan, if real action or observation differs from all branches
        elif agent.tree[real_action][real_observation] is not None:
            # Update the tree (prune)
            agent.tree = RootVNode.from_vnode(
                agent.tree[real_action][real_observation],
                agent.history)
            dd = TreeDebugger(agent.tree)

        else:
            raise ValueError("Unexpected state; child should not be None")

    def clear_agent(self):
        self._agent = None  # forget about current agent so that can plan for another agent.
        self._last_num_sims = -1

    def set_rollout_policy(self, rollout_policy):
        """
        set_rollout_policy(self, RolloutPolicy rollout_policy)
        Updates the rollout policy to the given one
        """
        self._rollout_policy = rollout_policy

    def _expand_vnode(self, vnode, history, state=None):

        for action in self._agent.valid_actions(state=state, history=history):
            if vnode[action] is None:
                history_action_node = QNode(self._num_visits_init,
                                            self._value_init)
                vnode[action] = history_action_node

        if self._action_prior is not None:
            # Using action prior; special values are set;
            for preference in \
                self._action_prior.get_preferred_actions(state, history):
                action, num_visits_init, value_init = preference
                history_action_node = QNode(num_visits_init,
                                            value_init)
                vnode[action] = history_action_node


    def _search(self):
        sims_count = 0
        time_taken = 0
        stop_by_sims = self._num_sims > 0

        if self._show_progress:
            if stop_by_sims:
                total = int(self._num_sims)
            else:
                total = self._planning_time
            pbar = tqdm(total=total)

        start_time = time.time()
        while True:
            ## Note: the tree node with () history will have
            ## the init belief given to the agent.
            state = self._agent.sample_belief()
            self._simulate(state, self._agent.history, self._agent.tree,
                           None, None, 0)
            sims_count +=1
            time_taken = time.time() - start_time

            if self._show_progress and sims_count % self._pbar_update_interval == 0:
                if stop_by_sims:
                    pbar.n = sims_count
                else:
                    pbar.n = time_taken
                pbar.refresh()

            if stop_by_sims:
                if sims_count >= self._num_sims:
                    break
            else:
                if time_taken > self._planning_time:
                    if self._show_progress:
                        pbar.n = self._planning_time
                        pbar.refresh()
                    break

        if self._show_progress:
            pbar.close()

        best_action = self._agent.tree.argmax()
        return best_action, time_taken, sims_count

    def _simulate(self,
                    state, history, root, parent,
                    observation, depth):
        if depth > self._max_depth:
            return 0
        if root is None:
            if self._agent.tree is None:
                root = self._VNode(agent=self._agent, root=True)
                self._agent.tree = root
                if self._agent.tree.history != self._agent.history:
                    raise ValueError("Unable to plan for the given history.")
            else:
                root = self._VNode()
            if parent is not None:
                parent[observation] = root
            self._expand_vnode(root, history, state=state)
            rollout_reward = self._rollout(state, history, root, depth)
            return rollout_reward
        action = self._ucb(root)
        robot_id = -114
        if isinstance(action, PickAction)  and action.obj_id not in state.object_states[robot_id].objects_found:
            ic ("Should not be happening - PO_UCT", action)
            exit()
        next_state, observation, reward, nsteps = sample_generative_model(self._agent, state, action)
        if nsteps == 0:
            # This indicates the provided action didn't lead to transition
            # Perhaps the action is not allowed to be performed for the given state
            # (for example, the state is not in the initiation set of the option,
            # or the state is a terminal state)
            return reward

        total_reward = reward + (self._discount_factor**nsteps)*self._simulate(next_state,
                                                                               history + ((action, observation),),
                                                                               root[action][observation],
                                                                               root[action],
                                                                               observation,
                                                                               depth+nsteps)
        root.num_visits += 1
        root[action].num_visits += 1
        root[action].value = root[action].value + (total_reward - root[action].value) / (root[action].num_visits)
        return total_reward

    def _rollout(self, state, history, root, depth):
        discount = 1.0
        total_discounted_reward = 0

        while depth < self._max_depth:
            action = self._rollout_policy.rollout(state, history)
            next_state, observation, reward, nsteps = sample_generative_model(self._agent, state, action)
            history = history + ((action, observation),)
            depth += nsteps
            total_discounted_reward += reward * discount
            discount *= (self._discount_factor**nsteps)
            state = next_state
        return total_discounted_reward

    def _ucb(self, root):
        """UCB1"""
        best_action, best_value = None, float('-inf')
        for action in root.children:
            if root[action].num_visits == 0:
                val = float('inf')
            else:
                val = root[action].value + \
                    self._exploration_const * math.sqrt(math.log(root.num_visits + 1) / root[action].num_visits)
            if val > best_value:
                best_action = action
                best_value = val
        return best_action

    def _sample_generative_model(self, state, action):
        '''
        (s', o, r) ~ G(s, a)
        '''

        if self._agent.transition_model is None:
            next_state, observation, reward = self._agent.generative_model.sample(state, action)
        else:
            next_state = self._agent.transition_model.sample(state, action)
            observation = self._agent.observation_model.sample(next_state, action)
            reward = self._agent.reward_model.sample(state, action, next_state)
        return next_state, observation, reward

    def _VNode(self, agent=None, root=False, **kwargs):
        """Returns a VNode with default values; The function naming makes it clear
        that this function is about creating a VNode object."""
        if root:
            return RootVNode(self._num_visits_init, self._agent.history)

        else:
            return VNode(self._num_visits_init)


def sample_generative_model(agent, state, action, discount_factor=1.0):
    """
    sample_generative_model(Agent agent, State state, Action action, float discount_factor=1.0)
    :math:`(s', o, r) \sim G(s, a)`

    If the agent has transition/observation models, a `black box` will be created
    based on these models (i.e. :math:`s'` and :math:`o` will be sampled according
    to these models).

    Args:
        agent (Agent): agent that supplies all the models
        state (State)
        action (Action)
        discount_factor (float): Defaults to 1.0; Only used when `action` is an :class:`Option`.

    Returns:
        tuple: :math:`(s', o, r, n_steps)`
    """

    if agent.transition_model is None:
        # |TODO: not tested|
        result = agent.generative_model.sample(state, action)
    else:
        result = sample_explict_models(agent.transition_model,
                                       agent.observation_model,
                                       agent.reward_model,
                                       state,
                                       action,
                                       discount_factor)
    return result


def sample_explict_models(T, O, R,
                            state, action, discount_factor=1.0):
    """
    sample_explict_models(TransitionModel T, ObservationModel O, RewardModel R, State state, Action action, float discount_factor=1.0)
    """
    nsteps = 0

    if isinstance(action, Option):
        # The action is an option; simulate a rollout of the option
        option = action
        if not option.initiation(state):
            # state is not in the initiation set of the option. This is
            # similar to the case when you are in a particular (e.g. terminal)
            # state and certain action cannot be performed, which will still
            # be added to the PO-MCTS tree because some other state with the
            # same history will allow this action. In this case, that certain
            # action will lead to no state change, no observation, and 0 reward,
            # because nothing happened.
            if O is not None:
                return state, None, 0, 0
            else:
                return state, 0, 0

        reward = 0
        step_discount_factor = 1.0
        while not option.termination(state):
            action = option.sample(state)
            next_state = T.sample(state, action)
            # For now, we don't care about intermediate observations (future work?).
            reward += step_discount_factor * R.sample(state, action, next_state)
            step_discount_factor *= discount_factor
            state = next_state
            nsteps += 1
        # sample observation at the end, where action is the last action.
        # (doesn't quite make sense to just use option as the action at this point.)
    else:
        next_state = T.sample(state, action)
        reward = R.sample(state, action, next_state)
        nsteps += 1
    if O is not None:
        observation = O.sample(next_state, action)
        return next_state, observation, reward, nsteps
    else:
        return next_state, reward, nsteps

class Option(Action):
    """An option is a temporally abstracted action that
    is defined by (I, pi, B), where I is a initiation set,
    pi is a policy, and B is a termination condition

    Described in `Between MDPs and semi-MDPs:
    A framework for temporal abstraction in reinforcement learning`
    """
    def initiation(self, state):
        """
        initiation(self, state)
        Returns True if the given parameters satisfy the initiation set"""
        raise NotImplementedError
    def termination(self, state):
        """termination(self, state)
        Returns a boolean of whether state satisfies the termination
        condition; Technically returning a float between 0 and 1 is also allowed."""
        raise NotImplementedError

    @property
    def policy(self):
        """Returns the policy model (PolicyModel) of this option."""
        raise NotImplementedError

    def sample(self, state):
        """
        sample(self, state)
        Samples an action from this option's policy.
        Convenience function; Can be overriden if don't
        feel like writing a PolicyModel class"""
        return self.policy.sample(state)

    def __eq__(self, other):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError