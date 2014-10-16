'''
Created on Oct 8, 2014

@author: Jan
'''
import unittest
import hierarchiconf



class Test(unittest.TestCase):
    def test_operational(self):
        Conf = hierarchiconf.Conf
        root_conf = Conf({'//'        : Conf({
                                              'seed':'global_seed',
                                              }),
                          'block1/seed': 'b1_seed',
                          'block2'     : Conf({
                                               'seed':'b2_seed',
                                               'dict_var': dict(imadict=True),
                                               })                                
                          })
        
        self.assertEqual(root_conf['seed'], 'global_seed')
        self.assertEqual(root_conf['block3/seed'], 'global_seed')
        self.assertEqual(root_conf['block1/seed'], 'b1_seed')
        b2_conf = root_conf.subconf('block2')
        self.assertEqual(b2_conf['seed'], 'b2_seed')
        self.assert_(isinstance(b2_conf['dict_var'], dict))
        b2_conf['var'] ='val'
        self.assertEqual(b2_conf['var'], 'val')
        self.assertEqual(root_conf['block2/var'], 'val')
        self.assertRaises(KeyError, lambda: root_conf['var'])
        self.assertRaises(KeyError, lambda: root_conf['block1/var'])
    
    def test_special(self):
        Conf = hierarchiconf.Conf
        root_conf = Conf({'//'        : Conf({
                                              'seed':'global_seed',
                                              }),
                          'block1/seed': 'b1_seed',
                          'block2'     : Conf({
                                               'seed':'b2_seed',
                                               'dict_var': dict(imadict=True),
                                               })                                
                          })
        self.assertEquals(root_conf['location'], '')
        self.assertEquals(root_conf.subconf('block1')['name'], 'block1')
        self.assertEquals(root_conf.subconf('block1')['location'], 'block1')
        self.assertEquals(root_conf.subconf('block1/sb1')['name'], 'sb1')
        self.assertEquals(root_conf.subconf('block1', 'sb2')['location'], 'block1/sb2')
        self.assertEquals(root_conf['block1/sb2/location'], 'block1/sb2')
        self.assertRaises(Exception, lambda : root_conf.__setitem__('block1/sb2/location','ttdd'))
    
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
                                             do_prefix_match=False, 
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
