from pomdp_problems.rearrange_pomdp.domain.state import MosOOState, ManipRobotState
import copy


def get_robot_state_from_full_state(state):
    #going through all objects and finding the one robot state object
    #Might need a better way in future but for now, this is what we got
    if not isinstance(state,MosOOState):
        raise TypeError(f"Expected MosOOState, got {type(state)}")
    for obj, obj_instance in state.object_states.items():
        if isinstance(obj_instance,ManipRobotState):
            robot_state = copy.deepcopy(obj_instance)
            return robot_state

    raise ValueError("Robot state not found in current state")