'''
Created on Nov 6, 2014

@author: Jan
'''

import numpy as np
import theano

import logging
logger = logging.getLogger(__name__)

from collections import namedtuple
from blocks.utils import sharedX
from pylearn2.utils.timing import log_timing

class AbstractStochasticTrainer(object):
    """
    A class for trainers that make updates after each minibatch.
    
    cost - expression for cost
    """
    def __init__(self, conf):
        self.conf = conf
    
    def init_train_fun(self, inputs, cost, parameters_grad_list, mandatory_updates, censor_updates_callback):
        pass
    
    #return 
    def do_one_iter(self, data):
        pass

class SharedVarStochasticTrainer(AbstractStochasticTrainer):
    def __init__(self, conf):
        super(SharedVarStochasticTrainer, self).__init__(conf)
        self.monitor_shared_values = []
        
    def init_train_fun(self, inputs, cost, parameters_grad_list, mandatory_updates, censor_updates_callback):
        #we will build two training funs: one which saves the data and gradients in shared vars and another one which 
        #does the actual update
        self.shared_inputs = []
        for x in inputs:
            dummy_placeholder = np.zeros( (2,)*x.ndim, dtype=x.dtype) 
            self.shared_inputs.append(theano.shared(dummy_placeholder,
                                                    name=x.name))
        self.params_grads = {}
        updates = list(mandatory_updates)
        self.shared_cost = sharedX(0.0, 'cost')
        updates.append((self.shared_cost, cost))
        for p,g in parameters_grad_list:
            shared_grad = sharedX(np.zeros(p.get_value(borrow=True).shape),
                                  name=p.name + '_grad')
            self.param_grads[p]=shared_grad
            updates.append((shared_grad, g))
            
        update_params, update_exprs = zip(*updates)
        # do we need to squeeze everything into theano.clone?? Is it faster or better - check
        cloned_expresssions = theano.clone(update_exprs, replace=zip(inputs, self.shared_inputs))
        
        with log_timing(logger, task='Compiling grad function'):
            self.comp_grad_fun = theano.function([], [], updates=zip(update_params, cloned_expresssions))
            
        with log_timing(logger, task='Compiling parameter update function'):
            self.init_update_fun()
    
    #override
    def init_update_fun(self):
        self.update_fun = None
        
    def do_one_iter(self, data):
        for x in self.shared_inputs:
            x.set_value(data[x.name], borrow=True)
        
        self.comp_grad_fun()
        self.update_fun()
        
        ret = []
        for sv in self.monitor_shared_values:
            ret.append((sv.name,sv.get_value()))
        return ret
