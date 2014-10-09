"""The blocks library for parameterized Theano ops"""
import numpy as np
from hierarchiconf import Conf

class Block(object):
    """Blocks are groups of bricks with a particular function.

    Bricks are disjoint subsets of the computational graph, which can
    potentially be reused several times throughout the graph. Blocks
    collect connected groups of bricks into groups. However, unlike bricks,
    blocks can be nested. This allows you to organise your model in a
    tree-like structure of nested blocks. In order to keep the
    tree-structure well-defined, blocks cannot be reused.

    Like bricks, blocks have `apply` methods which take Theano variables as
    input and output. Blocks can perform a lot of operations for you, like
    inferring what the input dimensions should be, or combining bricks in
    otherwise complicated ways.

    Within a block, the names of its children (blocks and bricks) need to
    be unique.

    Parameters
    ----------
    conf: hierarchical configuration

    Attributes
    ----------
    children : list of objects
        List of Block and Brick instances which belong to this class.
        Blocks can be expected to be children of this class alone, while
        bricks can also be part of other blocks.

    """
    def __init__(self, conf=Conf()):
        self.conf = conf
        self.name = conf['name']
        self.children = []
    
    def add_child(self, child):
        names = set(c.name for c in self.children)
        if child.name in names:
            raise Exception("Repeated name for child: %s" % (child.name,))
        self.children.append(child)
