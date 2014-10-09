"""Bricks module

This defines the basic interface of bricks.
"""
from abc import ABCMeta
import logging

import numpy as np
from theano import tensor

from blocks.utils import pack, reraise_as, sharedX, unpack

logging.basicConfig()
logger = logging.getLogger(__name__)

class Brick(object):
    """A brick encapsulates Theano operations with parameters.

    A Brick is a group of parameterized Theano operations. Bricks can be
    considered connected, disjoint subsets of nodes in Theano's
    computiational graph.

    Parameters
    ----------
    conf: a hierarchiconf.Conf object of parameters

    Attributes
    ----------
    params : list of Theano shared variables
        After calling the :meth:`allocate` method this attribute will be
        populated with the shared variables storing this brick's
        parameters.

    Notes
    -----
    Brick implementations *must* call the :meth:`__init__` constructor of
    their parent using `super(BlockImplementation,
    self).__init__(**kwargs)`.

    A brick can have any number of methods which apply the brick on Theano
    variables. These methods should be decorated with the
    :meth:`apply_method` decorator.
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, conf={}):
        self.conf = conf
        self.name = conf['name']

    def initialize_param(self, name, shape, **kwargs):
        param_conf = self.conf.subconf(name)
        param_conf.update(kwargs)
        value = None
        if param_conf['meta/reload']:
            saved_value = param_conf.get('value')
            if saved_value:
                assert saved_value.shape == shape
                value = saved_value
        if value is None:
            init_fcn = param_conf['init_fun']
            rng = param_conf.get('rng')
            if not rng:
                rng = np.random.RandomState(param_conf['seed'])
            value = init_fcn.generate(rng, shape)
        return sharedX(value, name=name)

    @staticmethod
    def apply_method(func):
        """Wraps methods that apply a brick to inputs in different ways.

        This decorator will provide some necessary pre- and post-processing
        of the Theano variables, such as tagging them with the brick that
        created them and naming them.

        Application methods will allocate the brick parameters with a call
        :meth:`allocate` if they have not been allocated already.

        Parameters
        ----------
        func : method
            A method which takes Theano variables as an input, and returns
            the output of the Brick

        Raises
        ------
        LazyInitializationError
            If parameters needed to perform the application of this brick
            have not been provided yet.

        """
        def wrapped_apply(self, *inputs, **kwargs):
            inputs = list(inputs)
            for i, inp in enumerate(inputs):
                inputs[i] = inp.copy()
            outputs = pack(func(self, *inputs, **kwargs))
            for output in outputs:
                # TODO Tag with dimensions, axes, etc. for error-checking
                output.tag.owner_brick = self
            return unpack(outputs)
        return wrapped_apply


class Linear(Brick):
    """A linear transformation with optional bias.

    Linear brick which applies a linear (affine) transformation by
    multiplying the input with a weight matrix. Optionally a bias is added.

    Configuration parameters
    ----------
    input_dim : int
        The dimension of the input. Required by :meth:`initialize`.
    output_dim : int
        The dimension of the output. Required by :meth:`initialize`.
    W/init : object
        A `NdarrayInitialization` instance which will be used by to
        initialize the weight matrix. Required by :meth:`initialize`.
    b/init : object, optional
        A `NdarrayInitialization` instance that will be used to initialize
        the biases. Required by :meth:`initialize` when `use_bias` is
        `True`.
    use_bias : bool
        Whether to use a bias.

    Notes
    -----

    A linear transformation with bias is a matrix multiplication followed
    by a vector summation.

    .. math:: f(\mathbf{x}) = \mathbf{W}\mathbf{x} + \mathbf{b}

    See also
    --------
    :class:`Brick`

    """
    def __init__(self, conf, **kwargs):
        super(Linear, self).__init__(conf, **kwargs)
        self.use_bias = conf.get('use_bias', True)
        self.W = self.init_param()
        if self.use_bias:
            W, b = self.params
            self.biases_init.initialize(b, self.rng, (self.output_dim,))
        else:
            W, = self.params
        self.weights_init.initialize(W, self.rng,
                                     (self.input_dim, self.output_dim))

    @Brick.apply_method
    def apply(self, inp):
        output = tensor.dot(inp, self.W)
        if self.use_bias:
            output += self.b
        return output

class Tanh(Brick):
    @Brick.apply_method
    def apply(self, inp):
        output = tensor.tanh(inp)
        return output


class Softmax(Brick):
    @Brick.apply_method
    def apply(self, inp):
        output = tensor.nnet.softmax(inp)
        return output


class Cost(Brick):
    pass


class CrossEntropy(Cost):
    @Brick.apply_method
    def apply(self, y, y_hat):
        cost = -(y * tensor.log(y_hat)).sum(axis=1).mean()
        return cost
