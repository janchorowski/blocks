import numpy as np
import theano
from blocks.utils import shared_for_expression

class Channel(object):
    """
    A Channel defines how to compute the value for a batch and how to accumulate partial 
    values over many batches to compute a summary for a whole dataset.
    """
    def __init__(self, conf, **kwargs):
        super(Channel, self).__init__()
        self.conf = conf
        self.accumulator_expressions = kwargs
    
    def get_single_batch_value(self):
        return None
    
    def get_accumulated_value(self, accumulator):
        return None
    
    def allocate_new_accumulator(self):
        accumulator = {}
        for k,v in self.accumulator_expressions.iteritems():
            accumulator[k]=shared_for_expression(v)
    
    def get_accumulator_initial_state_updates(self, accumulator):
        return []
    
    def get_accumulator_update(self, accumulator):
        return []
        
class Frac(Channel):
    def __init__(self, conf, numerator, denominator):
        super(Frac, self).__init__(conf, numerator=numerator, denominator=denominator)
            
    def get_single_batch_value(self):
        return self.accumulator_expressions['numerator']/self.accumulator_expressions['denominator']
    
    def get_accumulated_value(self, accumulator):
        return accumulator['numerator'].get_value()/accumulator['denominator'].get_value()
    
    def get_accumulator_initial_state_updates(self, accumulator):
        return [(accumulator['numerator'], 0.0),
                (accumulator['denominator'], 0.0),
                ]
    
    def get_accumulator_updates(self, accumulator):
        return [(accumulator['numerator'], accumulator['numerator'] + self.numerator),
                (accumulator['denominator'], accumulator['denominator'] + self.denominator),
                ]

class ParameterProperty(Channel):
    def __init__(self, conf, expression):
        super(ParameterProperty, self).__init__(conf, expression=expression)
        
    def get_single_batch_value(self):
        return self.accumulator_expressions['expression']
    
    def get_accumulated_value(self, accumulator):
        return accumulator['expression'].get_value()
    
    def reduce_accumulator(self, accumulator):
        return accumulator['expression']
    
    def get_accumulator_initial_state_updates(self, accumulator):
        return [(accumulator['expression'], self.accumulator_expressions['expression'])]
    
    def get_accumulator_updates(self, accumulator):
        return []

class SingleBatchMonitor(object):
    def __init__(self, monitors):
        self.monitors = monitors
        self.updates = []
        for m in self.monitors:
            val = m.get_single_batch_value()
            storage = theano.shared(np.zeros( (2,)*val.ndim,
                                                dtype=val.dtype),
                                    name=val.conf['location'])
            self.uptates.append((storage, val))
    
    def get_values(self, borrow=False):
        name_value_pairs = []
        for storage_var, unused_expression in self.updates:
            name_value_pairs.append(storage_var.name, storage_var.get_value(borrow=borrow)) #make sure we return a copy
        return name_value_pairs
             
        
class AggregatingMonitor(object):
    def __init__(self, monitors):
        self.monitors = []
        self.updates = []
        
        reset_updates = []
        for m in monitors:
            accumulator = m.allocate_new_accumulator()
            reset_updates.extend(m.get_accumulator_initial_state_updates(accumulator))
            self.updates.append(m.get_accumulator_updates(accumulator))
            self.monitors.append((m,accumulator))
        
        self.reset_function = theano.function([],[], updates=reset_updates, allow_input_downcast=True)

    def reset(self):
        self.reset_function()
    
    def get_values(self, borrow=False):
        name_value_pairs = []
        for m,accumulator in self.monitors:
            v = m.get_accumulated_value(accumulator)
            name_value_pairs.append(m.conf['location'], v)
        return name_value_pairs
            
    
class MonitorWriter(object):
    pass

class CSVMonitorWriter(MonitorWriter):
    def __init__(self, fname, append=True):
        self.file = None
        if append:
            mode='at'
        else:
            mode='wt'
        self.file = open(fname, mode)
        
        #detect if file is empty, and if yes write the header
        self.file.seek(0,2)
        if self.file.tell()==0L:
            self.file.write('num_examples_seen,context,name,value\n')
        
        self._num_examples_seen = None #used to trigger a flush whenever we start getting data for the next batch
    
    def append(self, num_examples_seen, context, name, value):
        if self._num_examples_seen!=num_examples_seen:
            self._num_examples_seen=num_examples_seen
            self.file.flush()
        self.write('%(num_examples_seen)s,%(context)s,%(name)s,%(value)\n' % locals())
        
    def append_many(self, num_examples_seen, context, name_value_pairs):
        if self._num_examples_seen!=num_examples_seen:
            self._num_examples_seen=num_examples_seen
            self.file.flush()
        for name, value in name_value_pairs:
            self.write('%(num_examples_seen)s,%(context)s,%(name)s,%(value)\n' % locals())
    
    def close(self):
        if self.file:
            self.file.close()
            self.file = None

class ConsoleWriter(MonitorWriter):
    def __init__(self):
        self._num_examples_seen = None #used to trigger a flush whenever we start getting data for the next batch
        self._name_values = []
        
    def append(self, num_examples_seen, context, name, value):
        self.append_many(num_examples_seen, context, [()])
              
    def append_many(self, num_examples_seen, context, name_value_pairs):
        if self._num_examples_seen!=num_examples_seen:
            self._num_examples_seen=num_examples_seen
            message = ['%(num_examples_seen)s,%(context)s:' % locals()]
            for name_val in self._name_values:
                message.append('%s=%.3g' % name_val)
            print ' '.join(message)
            self._name_values = name_value_pairs
        else:
            self._name_values.extend(name_value_pairs)
