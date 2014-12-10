
import logging
from blocks.monitor import SingleBatchMonitor, ConsoleWriter
from blocks.bricks import Brick
logger = logging.getLogger(__name__)

import theano

from blocks.utils import collect_tag, collect_parameters, attach_context
from pylearn2.utils.timing import log_timing

class Model(Brick):
    def __init__(self, conf, inputs=None, top_brick=None):
        super(Model, self).__init__(conf)
        self.inputs = inputs
    
    def get_unregularized_cost(self, context):
        inputs = attach_context(context, self.inputs)
        return self._get_unregularized_cost(self, inputs)
    
    def _get_unregularized_cost(self, inputs):
        raise NotImplementedError()
        
    def get_monitors(self, unregularized_cost):
        return sum(collect_tag(outputs=[self.unregularized_cost], tag='monitors'),[])
    
    def get_regularizations(self, unregularized_cost):
        return sum(collect_tag(outputs=[self.unregularized_cost], tag='regularizations'), [])
    
    def get_updates(self, unregularized_cost):
        return sum(collect_tag(outputs=[self.unregularized_cost], tag='updates'), [])
    
    def get_parameters(self, unregularized_cost):
        return collect_parameters(outputs=[unregularized_cost])
    
    def get_gradients(self, cost, parameters):
        return theano.grad(cost, parameters)
    
    def censor_updates(self, updates):
        censored_updates = []
        for shared_var, new_value in updates:
            censored_value = new_value
            if hasattr(shared_var, 'conf'):
                conf = shared_var.conf
                #TODO here: constrain column norm, constrain row norm
            censored_updates.append((shared_var, censored_value))
        return censored_updates
       
    def validate(self, dataset):
        pass

class StochasticMainLoop(object):
    def __init__(self, conf, model_constructor):
        self.conf = conf
        self.model = model_constructor(conf.subconf('model'))
        self.monitor_writers = [ConsoleWriter()]
    
    def monitor(self, num_examples_seen, context, monitors):
        for mw in self.monitor_writers:
            for nv in name_values:
                mw.append_many(num_examples_seen, context, nv)
                
    def run_main_loop(self, dataset):
        model = self.model
        
        trainer_conf = self.conf.subconf('trainer')
        trainer = trainer_conf['class'](trainer_conf)
        
        unregularized_cost = model.get_unregularized_cost(self.conf.subconf('train_context'))
        
        with log_timing(logger, "Collecting parameters, monitors and regularizations from the graph"):
            parameters = model.get_parameters(unregularized_cost)
            monitors = model.get_monitors(unregularized_cost)
            regularizations = model.get_regularizations(unregularized_cost)
            updates = model.get_updates(unregularized_cost)
        
        tot_cost = unregularized_cost
        for reg in regularizations:
            tot_cost += reg.reduce() #XXX: should we multiply here by reg.conf['scale']?
            monitors.append(reg)
        
        with log_timing(logger, "Computing gradients"):
            param_grads = model.get_gradients(tot_cost, parameters)
            parameters_grad_list = zip(parameters, param_grads)
        
        batch_monitor = SingleBatchMonitor(monitors)
        
        trainer.init_train_fun(model.inputs, tot_cost, parameters_grad_list, 
                               mandatory_updates=updates + batch_monitor.updates, 
                               censor_updates_callback=model.censor_updates)
        
        for i in xrange(trainer.conf['num_epochs']):
            for batch in dataset.iterate():
                additional_monitors = trainer.do_one_iter(batch) 
            self.monitor(num_examples_seen=-1, context='training', monitors=(batch_monitor.get_values(), additional_monitors))
