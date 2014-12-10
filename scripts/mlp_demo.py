'''
Created on Oct 17, 2014

@author: Jan
'''

from blocks import bricks, initialization, model
from hierarchiconf import Conf

import theano.tensor as TT
from theano import printing
from blocks.monitor import Frac
from blocks.dataset import PL2DataSetAdapter
from pylearn2.datasets import mnist

def default_conf(): 
    return Conf({
        '//': Conf({
            'seed': 1234,
            
            #initlaization
            'W.*/init_fun' : initialization.Uniform(width=1e-2),
            'b.*/init_fun' : initialization.Constant(0.0),
        
            #default brick structure:
            'use_bias' : True,
        
            #meta-configuration
            'meta' : Conf({
                'reload' : True
            }),
        }),
    })

class DemoModel(model.Model):
    def __init__(self, conf, data):
        super(DemoModel, self).__init__(conf)
        
        self.X = TT.matrix('X')
        self.T = TT.lvector('T')
        self.inputs = [self.X, self.T]
        
        batch = data.peek()
        conf['input_dim'] = batch['X'].shape[1]
        conf['layers'].append({'layer_class':bricks.SoftmaxLayer, 'output_dim':data.get_num_labels()})
        self.top_brick = bricks.MLP(conf)
                
    def _get_unregularized_cost(self, X,T):
        Y = self.mlp.apply(X)
        T_pred = TT.argmax(Y, axis=1)
        
        err_rate = Frac(self.conf.subconf('monitors/err_rate'), 
                        numerator=(T!=T_pred).sum(), denominator=Y.shape[0])
        
        sum_log_like = -TT.sum(TT.log(Y)[TT.arange(T.shape[0]), T])
        
        self.add_monitor(sum_log_like, err_rate)
        
        log_likelihood = Frac(self.conf.subconf('cost'),
                              numerator=sum_log_like,
                              denominator=Y.shape[0])
        
        return log_likelihood
        
if __name__=='__main__':
    #{'layer_class': bricks.MaxOutLayer, 'output_dim':10, 'num_filters':5}
    conf = default_conf()
    conf['model'] = Conf({'layers':[25,
                                    25
                                    ], 
                          })
    
    train_data = PL2DataSetAdapter(conf.subconf('train_data'), mnist.MNIST('train'))
    model = DemoModel(conf.subconf('model'), data=train_data)
    