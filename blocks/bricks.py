"""Bricks module

This defines the basic interface of bricks.
"""
from abc import ABCMeta
import logging

import numpy as np
from theano import tensor

from blocks.utils import pack, reraise_as, sharedX, unpack
from hierarchiconf import Conf

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
        ret = sharedX(value, name=param_conf['location'])
        ret.tag.conf = param_conf
        ret.tag.trainable = True
        return ret

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
    
    def add_monitor(self, expression, monitor):
        monitors = getattr(expression.tag,'monitors',[])
        monitors.append(monitor)
        expression.tag.monitors=monitors
    
    def add_regularization(self, expression, regularization):
        regularizations = getattr(expression.tag,'regularization',[])
        regularizations.append(regularization)
        expression.tag.regularizations=regularizations

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

class LinearAndNonlinearLayer(Brick):
    def __init__(self, conf, nonlinearity_class, **kwargs):
        super(LinearAndNonlinearLayer, self).__init__(conf, **kwargs)
        #Question: do we want to grab the sub-conf here too?
        self.linear = Linear(conf)
        self.nonlinearity = nonlinearity_class(conf)
        
    @Brick.apply_method
    def apply(self, inp):
        output = self.linear.apply(inp)
        output = self.nonlinearity.apply(output)
        return output
    
class TanhLayer(LinearAndNonlinearLayer):
    def __init__(self, conf, **kwargs):
        super(TanhLayer, self).__init__(conf, nonlinearity_class=Tanh, **kwargs)

class SoftmaxLayer(LinearAndNonlinearLayer):
    def __init__(self, conf, **kwargs):
        super(SoftmaxLayer, self).__init__(conf, nonlinearity_class=Softmax, **kwargs)
   
class MaxOutLayer(Brick):
    def __init__(self, conf, **kwargs):
        super(MaxOutLayer, self).__init__(conf, **kwargs)
        self.use_bias = conf.get('use_bias', True)
        self.num_filters = conf['num_filters']
        self.W = self.initialize_param('W', (conf['input_dim'], conf['output_dim']*self.num_filters))
        if self.use_bias:
            self.b = self.initialize_param('b', (conf['output_dim']*self.num_filters, ))
    
    @Brick.apply_method
    def apply(self, inp):
        output = tensor.dot(inp, self.W)
        if self.use_bias:
            output += self.b
        output = output.reshape((output.shape[0], self.conf['output_dim'], self.num_filters))
        output = tensor.max(output, 2)
        return output
    
class MLP(Brick):
    def __init__(self, conf, **kwargs):
        super(MLP, self).__init__(conf, **kwargs)
        layer_confs = self.conf['layers']
        self.layers = []
        input_dim = conf['input_dim']
        #
        # This code is somewhat complicated, because we want to:
        # - allow easy configuration of layer sizes by just passing a list of ints
        # - allow more per-layer configuration by just passing a list of dicts/Conf objects
        # - allow easy setting of layer defaults by using 'layer_.*' wildcard matches
        #
        for layer_num, layer_conf_def in enumerate(layer_confs):
            #make sure that layer_conf_def is a Conf:
            # if it is dict, use it to make a new Conf
            # if it is int, assume it is the size of the new layer
            if isinstance(layer_conf_def, dict):
                layer_conf_def = Conf(layer_conf_def)
            if not isinstance(layer_conf_def, Conf):
                layer_conf_def=Conf({'output_dim':layer_conf_def})
            
            layer_conf_def['input_dim'] = input_dim
            input_dim = layer_conf_def['output_dim'] #for next layer
            
            if layer_num+1 == len(layer_confs):
                layer_name='layer_out'
            else:
                layer_name='layer_%d' % (layer_num, )
            
            conf[layer_name] = layer_conf_def #add configuration for this layer to the main Conf object
            layer_conf = conf.subconf(layer_name)
            layer_class = layer_conf.get('layer_class', TanhLayer)
            layer = layer_class(layer_conf)
            self.layers.append(layer)
    
    @Brick.apply_method
    def apply(self, x):
        for layer in self.layers:
            x = layer.apply(x)
        return x

class Cost(Brick):
    pass


class CrossEntropy(Cost):
    @Brick.apply_method
    def apply(self, y, y_hat):
        cost = -(y * tensor.log(y_hat)).sum(axis=1).mean()
        return cost
