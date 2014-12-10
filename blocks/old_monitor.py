import numpy as np
import theano

import abc

class Channel(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, conf, **kwargs):
        self.conf = conf
        self.expressions = kwargs #a dict of what to evaluate for each mini_batch
        self.data_dependent = True #do expressions depend on the data? If not we can run them once during validation
    
    @abc.abstractmethod
    def _reduce(self, expression_values):
        """
        Reduce a list of dicts, whose values correspond to expressions passed in kwargs to init into a single value. 
        """
        assert self.expressions=={}
        return None
    
    def reduce(self, expression_values=None):
        if expression_values:
            return self._reduce(expression_values)
        else:
            return self._reduce([self.expressions])

class Mean(Channel):
    def __init__(self, conf, numerator, denominator):
        super(Mean, self).__init__(conf, numerator=numerator, denominator=denominator)
        
    def _reduce(self, expression_values):
        ds = 0.0
        ns = 0.0
        for ev in expression_values:
            ds += ev['denominator']
            ns += ev['numerator']
        return ns/ds
    
class ParameterProperty(Channel):
    def __init__(self, conf, expression):
        super(ParameterProperty, self).__init__(conf, expression=expression)
        self.data_dependent = False
        
    def _reduce(self, expression_values):
        ret = expression_values[0]['expression']
        for ev in expression_values[1:]:
            assert ret==ev['expression'], 'The ParameterProperty expression cannot change batch-to-batch as it can only depend on parameters, not on inputs'
        return ret

class SingleBatchMonitor(object):
    def __init__(self, monitors):
        self.monitors = monitors
        self.updates = []
        for m in self.monitors:
            val = m.reduce()
            storage = theano.shared(np.zeros( (2,)*val.ndim,
                                                dtype=val.dtype),
                                    name=val.conf['location'])
            self.uptates.append((storage, val))
    
    def get_values(self, borrow=False):
        name_value_pairs = []
        for storage_var, unused_expression in self.updates:
            name_value_pairs.append(storage_var.name, storage_var.get_value(borrow=borrow)) #make sure we return a copy
        return name_value_pairs
            
    def report(self, monitor_writer, num_examples_seen, context='train'):
        monitor_writer.append_many(num_examples_seen, context, self.get_values())
        
class AggregatingMonitor(object):
    def __init__(self, monitors):
        self.monitors = []
        self.updates = []
        for m in monitors:
            storage = {}
            for expr_name, expr_val in m.expressions 
            val = m
            storage = theano.shared(np.zeros( (2,)*val.ndim,
                                                dtype=val.dtype),
                                    name=val.conf['location'])
            self.uptates.append((storage, val))
    
    
    
    def get_values(self, borrow=False):
        name_value_pairs = []
        for storage_var, unused_expression in self.updates:
            name_value_pairs.append(storage_var.name, storage_var.get_value(borrow=borrow)) #make sure we return a copy
        return name_value_pairs
            
    def report(self, monitor_writer, num_examples_seen, context='train'):
        monitor_writer.append_many(num_examples_seen, context, self.get_values()) 
    
class WholeDatasetMonitor(object):
    pass

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
