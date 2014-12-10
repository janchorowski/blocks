'''
Created on Nov 11, 2014

@author: Jan
'''

class PL2DataSetAdapter():
    def __init__(self, conf, dataset):
        self.conf = conf
        self.pl2data = dataset
        
    def peek(self):
        bs = self.conf.get('bs', 10)
        X,T = self.pl2data.get_batch_design(bs, True)
        return dict(X=X, T=T)
    
    def get_iterator(self):
        bs = self.conf.get('bs', 10)
        mode = self.conf.get('mode', 'uniform')
        iter = self.pl2data.iterator(mode, bs, targets=True)
        for (X,T) in iter:
            yield dict(X=X, T=T)

    def get_num_labels(self):
        return self.pl2data.y_labels
        