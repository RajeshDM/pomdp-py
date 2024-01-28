from pomdp_py.utils import typ

class TreeNode:
    def __init__(self):
        self.children = {}
    def __getitem__(self, key):
        return self.children.get(key, None)
    def __setitem__(self, key, value):
        self.children[key] = value
    def __contains__(self, key):
        return key in self.children

class QNode(TreeNode):
    def __init__(self, num_visits, value):
        """
        `history_action`: a tuple ((a,o),(a,o),...(a,)). This is only
            used for computing hashses
        """
        self.num_visits = num_visits
        self.value = value
        self.children = {}  # o -> VNode
    def __str__(self):
        return typ.red("QNode") + "(%.3f, %.3f | %s)" % (self.num_visits,
                                                         self.value,
                                                         str(self.children.keys()))
    def __repr__(self):
        return self.__str__()

class VNode(TreeNode):
    def __init__(self, num_visits, **kwargs):
        self.num_visits = num_visits
        self.children = {}  # a -> QNode
    def __str__(self):
        return typ.green("VNode") + "(%.3f, %.3f | %s)" % (self.num_visits,
                                                           self.value,
                                                           str(self.children.keys()))
    def __repr__(self):
        return self.__str__()

    def print_children_value(self):
        for action in self.children:
            print("   action %s: %.3f" % (str(action), self[action].value))

    def argmax(self):
        """argmax(VNode self)
        Returns the action of the child with highest value"""
        #cdef Action action, best_action
        #cdef float best_value = float("-inf")
        best_value = float("-inf")
        best_action = None
        for action in self.children:
            if self[action].value > best_value:
                best_action = action
                best_value = self[action].value
        return best_action

    @property
    def value(self):
        best_action = max(self.children, key=lambda action: self.children[action].value)
        return self.children[best_action].value


class RootVNode(VNode):
    def __init__(self, num_visits, history):
        VNode.__init__(self, num_visits)
        self.history = history
    @classmethod
    def from_vnode(cls, vnode, history):
        """from_vnode(cls, vnode, history)"""
        rootnode = RootVNode(vnode.num_visits, history)
        rootnode.children = vnode.children
        return rootnode