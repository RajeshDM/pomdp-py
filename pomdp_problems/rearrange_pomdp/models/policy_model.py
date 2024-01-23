"""Policy model for 2D Multi-Object Search domain. 
It is optional for the agent to be equipped with an occupancy
grid map of the environment.
"""

import pomdp_py
import random
from pomdp_problems.rearrange_pomdp.domain.action import *
from pomdp_problems.rearrange_pomdp.domain.state import *
from pomdp_problems.rearrange_pomdp.models.utils import get_robot_state_from_full_state

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
        """note 2: Pick can only happen after find - NO LONGER THE CASE"""
        can_find = False
        can_pick = True
        can_place = False
        if history is not None and len(history) > 1:
            # last action
            last_action = history[-1][0]
            if isinstance(last_action, LookAction):
                can_find = True
            #if isinstance(last_action, FindAction) or \
            #    isinstance(last_action,PickAction):
            #    can_pick = True
        find_action = set({Find}) if can_find else set({})

        pick_actions = set({})
        place_actions = set({})
        if can_pick :
            robot_state = get_robot_state_from_full_state(state)
            for obj, obj_instance in state.object_states.items():
                if isinstance(obj_instance, ManipObjectState):
                    if obj in set(robot_state.objects_found) :
                        if obj_instance.is_held is False :
                            pick_actions.add(PickAction(obj_instance.objid))

                    # TODO - p.1 - DONE 
                    # Will ADD this back when put down action is implemented
                    # UNTIL THEN, we will assume robot can carry multiple items
                    #elif obj_instance.is_held is True :
                    #    can_place = True 
                    #    can_pick = False
                    #    break
                elif isinstance(obj_instance,ManipRobotState):
                    if len(obj_instance.holding) >= obj_instance.carry_cap :    
                        can_place = True 
                        can_pick = False
                        break

        if can_pick is False :
            pick_actions = set({})

        can_place = False
        if can_place is True :
            place_actions = set({PlaceAction()})

        available_actions = {Look} | find_action | pick_actions
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

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]
