"""Policy model for 2D Multi-Object Search domain. 
It is optional for the agent to be equipped with an occupancy
grid map of the environment.
"""

import pomdp_py
import random
from pomdp_problems.rearrange_pomdp.domain.action import *

class PolicyModel(pomdp_py.RolloutPolicy):
    """Simple policy model. All actions are possible at any state."""

    def __init__(self, robot_id, grid_map=None):
        """FindAction can only be taken after LookAction"""
        self.robot_id = robot_id
        self._grid_map = grid_map

    def sample(self, state, **kwargs):
        return random.sample(self._get_all_actions(**kwargs), 1)[0]
    
    def probability(self, action, state, **kwargs):
        raise NotImplementedError

    def argmax(self, state, **kwargs):
        """Returns the most likely action"""
        raise NotImplementedError

    def get_all_actions(self, state=None, history=None):
        """note: find can only happen after look."""
        """note 2: Pick can only happen after find"""
        can_find = False
        can_pick = False
        if history is not None and len(history) > 1:
            # last action
            last_action = history[-1][0]
            if isinstance(last_action, LookAction):
                can_find = True
            if isinstance(last_action, FindAction):
                can_pick = True
        find_action = set({Find}) if can_find else set({})
        pick_action = set({Pick}) if can_pick else set({})

        available_actions = {Look} | find_action | pick_action
        #available_actions = {Look} | find_action 

        if state is None :
            available_actions = available_actions | ALL_MOTION_ACTIONS
        else: 
            if self._grid_map is not None:
                valid_motions =\
                    self._grid_map.valid_motions(self.robot_id,
                                                 state.pose(self.robot_id),
                                                 ALL_MOTION_ACTIONS)
                available_actions = available_actions | valid_motions
            else :
                available_actions = available_actions | ALL_MOTION_ACTIONS

        return available_actions

        #REMOVE AFTER TESTING
        '''
        if state is None:
            return ALL_MOTION_ACTIONS | {Look} | find_action | pick_action
        else:
            if self._grid_map is not None:
                valid_motions =\
                    self._grid_map.valid_motions(self.robot_id,
                                                 state.pose(self.robot_id),
                                                 ALL_MOTION_ACTIONS)
                return valid_motions | {Look} | find_action | pick_action
            else:
                return ALL_MOTION_ACTIONS | {Look} | find_action | pick_action
        '''

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]
