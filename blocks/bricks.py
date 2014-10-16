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
    def __init__(self, conf={}):
        self.conf = conf
        
    @property
    def name(self):
        return self.conf['name']

    @property
    def location(self):
        """
        Location of this brick in the creation hierarchy. 
        """
        return self.conf['location']


    def initialize_param(self, name, shape, **kwargs):
        param_conf = self.conf.subconf(name)
        param_conf.update(kwargs)
        value = None
        if param_conf['meta/reload']:
            saved_value = param_conf.get('value')
            if saved_value:
                assert saved_value.shape == shape
                value = saved_value
                logger.info('Reloaded value for %s.', param_conf['location'])
        if value is None:
            init_fcn = param_conf['init_fun']
            rng = param_conf.get('rng')
            if not rng:
                rng = np.random.RandomState(param_conf['seed'])
            value = init_fcn.generate(rng, shape)
        return sharedX(value, name=name)

    @staticmethod
    def apply_method(func):
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
    def __init__(self, conf, **kwargs):
        super(Linear, self).__init__(conf, **kwargs)
        self.use_bias = conf.get('use_bias', True)
        self.W = self.initialize_param('W', (conf['input_dim'], conf['output_dim']))
        if self.use_bias:
            self.b = self.initialize_param('b', (conf['output_dim'],))

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
