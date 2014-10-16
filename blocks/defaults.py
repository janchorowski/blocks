'''
Created on Oct 9, 2014

@author: chorows
'''

from hierarchiconf import Conf
from blocks import initialization


default_conf = Conf({'//': #Least priority globals
    Conf({
        #reproducibility:
        'seed'     : [9,10,2014], 
          
        #initlaization
        'W.*/init_fun' : initialization.Uniform(width=1e-2),
        'B.*/init_fun' : initialization.Constant(0.0),
        
        #default brick structure:
        'use_bias' : True,
        
        #meta-configuration
        'meta' : Conf({
            'reload' : True
            })
        })})
