import cPickle as pickle

from blocks.utils import SkipCache, cached
from nose.tools import assert_raises


class A(SkipCache):
    def __init__(self, val):
        self.v = val

    @cached
    def val(self):
        return self.v

    @cached
    def val1(self):
        return self.v + 1

    @cached
    def val2(self, v):
        return self.v + v


class B():
    def __init__(self, val):
        self.v = val

    @cached
    def val(self):
        return self.v


def test_cache():
    a = A(10)
    assert a.val() == 10
    assert a.val1() == 11
    a.v = 20
    assert a.val() == 10
    assert a.val1() == 11

    del a.__cache__
    assert a.val() == 20
    assert a.val1() == 21

    assert A(10).val() == 10
    assert A(20).val() == 20
    assert A(30).val() == 30

    a_val = A(10).val
    b_val = A(20).val
    c_val = A(30).val

    assert a_val() == 10
    assert b_val() == 20
    assert c_val() == 30


def test_class_asserts():
    b = B(10)
    assert_raises(Exception, lambda: b.val())

    a = A(10)
    assert_raises(Exception, lambda: a.val2(10))


def test_pickle():
    a = A(10)
    assert a.val() == 10
    a.v = 20
    assert a.val() == 10

    a_pickled = pickle.dumps(a)
    aa = pickle.loads(a_pickled)
    assert aa.val() == 20
