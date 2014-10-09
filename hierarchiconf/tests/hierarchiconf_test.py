'''
Created on Oct 8, 2014

@author: Jan
'''
import unittest
import hierarchiconf

Conf = hierarchiconf.Conf

sample_conf = Conf({'//': Conf({'seed':[1,2,3],
                                }),
                    'block1/seed': [4,5,6], 
                    'block2' : Conf({
                        'seed':'b2seed',
                        'dvar':dict(imadict=True),             
                        })                                
                    })

class Test(unittest.TestCase):
    def test_split_to_parts(self):
        sp = hierarchiconf._split_to_parts_valid
        self.assertEqual(sp(['a','b','c']), ['a','b','c'])
        self.assertEqual(sp(['a//q/w','b','c']), ['a','//','q','w','b','c'])
        self.assertEqual(sp(['a//q/w','b','c/d/e//f']), ['a','//','q','w','b','c','d','e','//','f'])
    
    def test_key_path_join(self):
        j = hierarchiconf._selector_path_join_valid
        self.assertEqual(j('a'), 'a')
        self.assertEqual(j('a','b'), 'a/b')
        self.assertEqual(j('a','b/c'), 'a/b/c')
        self.assertEqual(j('a','b//c'), 'a/b//c')
        self.assertEqual(j('//','b//c'), '//b//c')
        self.assertEqual(j('//','//'), '//')
        
    def test_match(self):
        m = lambda s,p: hierarchiconf._match(hierarchiconf._split_to_parts_valid(s),
                                             hierarchiconf._split_to_parts_valid(p),
                                             allow_prefix_match=False, 
                                             specificity=hierarchiconf._base_specificity)
        self.assertTrue(m(['lrate'], ['lrate']))
        self.assertTrue(m(['.*/lrate'], ['aa/lrate']))
        self.assertFalse(m(['.*/lrate'], ['aa/bb/lrate']))
        self.assertTrue(m(['//lrate'], ['aa/lrate']))
        self.assertTrue(m(['//lrate'], ['aa/bb/lrate']))
        self.assertFalse(m(['aa'], ['bb']))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_split_to_parts']
    unittest.main()
