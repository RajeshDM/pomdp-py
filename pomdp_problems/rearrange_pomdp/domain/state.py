"""Defines the State for the 2D Multi-Object Search domain;

Origin: Multi-Object Search using Object-Oriented POMDPs (ICRA 2019)
(extensions: action space changes, different sensor model, gridworld instead of
topological graph)

Description: Multi-Object Search in a 2D grid world.

State space: 

    :math:`S_1 \\times S_2 \\times ... S_n \\times S_r`
    where :math:`S_i (1\leq i\leq n)` is the object state, with attribute
    "pose" :math:`(x,y)` and Sr is the state of the robot, with attribute
    "pose" :math:`(x,y)` and "objects_found" (set).
"""

import pomdp_py
import math

###### States ######
class ObjectState(pomdp_py.ObjectState):
    def __init__(self, objid, objclass, pose):
        if objclass != "obstacle" and objclass != "target":
            raise ValueError("Only allow object class to be"\
                             "either 'target' or 'obstacle'."
                             "Got %s" % objclass)
        super().__init__(objclass, {"pose":pose, "id":objid})
    def __str__(self):
        return 'ObjectState(%s,%s)' % (str(self.objclass), str(self.pose))
    @property
    def pose(self):
        return self.attributes['pose']
    @property
    def objid(self):
        return self.attributes['id']

###### Manipulation capable States ######
class ManipObjectState(pomdp_py.ObjectState):
    def __init__(self, objid, objclass, pose, other_attrs = None):
        if objclass != "obstacle" and objclass != "target":
            raise ValueError("Only allow object class to be"\
                             "either 'target' or 'obstacle'."
                             "Got %s" % objclass)

        all_attrs = {"pose":pose, "id":objid}

        if other_attrs is None or "is_held" not in other_attrs:
            all_attrs.update({"is_held" : False})
        
        if other_attrs != None :
            all_attrs.update(other_attrs)

        super().__init__(objclass, all_attrs)
    def __str__(self):
        return 'ManipObjectState(%s,%s,%s)' % (str(self.objclass), str(self.pose), str(self.is_held))
    @property
    def pose(self):
        return self.attributes['pose']
    @pose.setter
    def pose(self,value):
        self.attributes['pose'] = value

    @property
    def objid(self):
        return self.attributes['id']

    @property
    def is_held(self):
        return self.attributes['is_held']
    @is_held.setter
    def is_held(self, is_held_val):
        self.attributes['is_held'] = is_held_val

class ManipRobotState(pomdp_py.ObjectState):
    def __init__(self, robot_id, pose, objects_found, camera_direction):
        """Note: camera_direction is None unless the robot is looking at a direction,
        in which case camera_direction is the string e.g. look+x, or 'look'"""
        super().__init__("robot", {"id":robot_id,
                                   "pose":pose,  # x,y,th
                                   "objects_found": objects_found,
                                   "camera_direction": camera_direction,
                                   "objects_picked" : 0, 
                                   "is_holding": False})
    def __str__(self):
        return 'ManipRobotState(%s,%s|%s|%s|%s)' % (str(self.objclass), str(self.pose), \
                                         str(self.objects_found), str(self.objects_picked),\
                                        str(self.is_holding))
    def __repr__(self):
        return str(self)
    @property
    def pose(self):
        return self.attributes['pose']

    @property
    def objid(self):
        return self.attributes['id']

    @property
    def robot_pose(self):
        return self.attributes['pose']

    @property
    def objects_found(self):
        return self.attributes['objects_found']
    @property
    def objects_picked(self):
        return self.attributes['objects_picked']
    @property
    def camera_direction(self):
        return self.attributes['camera_direction']

    # Added this property but not using it atm - 
    # It is being currently determined if any of the objects are being held
    @property
    def is_holding(self):
        return self.attributes['is_holding']
    

class MosOOState(pomdp_py.OOState):
    def __init__(self, object_states):
        super().__init__(object_states)
    def object_pose(self, objid):
        return self.object_states[objid]["pose"]
    def pose(self, objid):
        return self.object_pose(objid)
    def is_held(self,objid):
        return self.object_states[objid]['is_held']
    @property
    def object_poses(self):
        return {objid:self.object_states[objid]['pose']
                for objid in self.object_states}
    def __str__(self):
        return 'MosOOState%s' % (str(self.object_states))
    def __repr__(self):
        return str(self)
