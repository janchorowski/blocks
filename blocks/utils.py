import sys

import numpy as np

import theano
from theano.compile.sharedvalue import SharedVariable


def pack(arg):
    """Pack variables into a list.

    Parameters
    ----------
    arg : object
        Either a list or tuple, or any other Python object. Lists will be
        returned as is, and tuples will be cast to lists. Any other
        variable will be returned in a singleton list.

    Returns
    -------
    list
        List containing the arguments

    """
    if isinstance(arg, (list, tuple)):
        return list(arg)
    else:
        return [arg]


def unpack(arg):
    """Unpack variables from a list or tuple.

    Parameters
    ----------
    arg : object
        Either a list or tuple, or any other Python object. If passed a
        list or tuple of length one, the only element of that list will
        be returned. If passed a tuple of length greater than one, it
        will be cast to a list before returning. Any other variable
        will be returned as is.

    Returns
    -------
    object
        A list of length greater than one, or any other Python object
        except tuple.

    """
    if isinstance(arg, (list, tuple)):
        if len(arg) == 1:
            return arg[0]
        else:
            return list(arg)
    else:
        return arg


def sharedX(value, name=None, borrow=False, dtype=None):
    """Transform a value into a shared variable of type floatX.

    Parameters
    ----------
    value : array_like
        The value to associate with the Theano shared.
    name : str, optional
        The name for the shared varaible. Defaults to `None`.
    borrow : bool, optional
        If set to True, the given `value` will not be copied if possible.
        This can save memory and speed. Defaults to False.
    dtype : str, optional
        The `dtype` of the shared variable. Default value is
        `theano.config.floatX`.

    Returns
    -------
    theano.compile.SharedVariable
        A Theano shared variable with the requested value and `dtype`.

    """

    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
                         name=name,
                         borrow=borrow)

def shared_for_expression(expression, name):
    return theano.shared(np.zeros( (2,)*expression.ndim,
                                   dtype=expression.dtype),
                         name=name)

def reraise_as(new_exc):
    """
    Parameters
    ----------
    new_exc : Exception isinstance
        The new error to be raised e.g. (ValueError("New message"))
        or a string that will be prepended to the original exception
        message

    Notes
    -----
    Note that when reraising exceptions, the arguments of the original
    exception are cast to strings and appended to the error message. If
    you want to retain the original exception arguments, please use:

    >>> try:
    ...     1 / 0
    ... except Exception as e:
    ...     reraise_as(Exception("Extra information", *e.args))
    Traceback (most recent call last):
      ...
    Exception: 'Extra information, ...

    Examples
    --------
    >>> class NewException(Exception):
    ...     def __init__(self, message):
    ...         super(NewException, self).__init__(message)
    >>> try:
    ...     do_something_crazy()
    ... except Exception:
    ...     reraise_as(NewException("Informative message"))
    Traceback (most recent call last):
      ...
    NewException: Informative message ...

    """
    orig_exc_type, orig_exc_value, orig_exc_traceback = sys.exc_info()

    if isinstance(new_exc, basestring):
        new_exc = orig_exc_type(new_exc)

    if hasattr(new_exc, 'args'):
        if len(new_exc.args) > 0:
            # We add all the arguments to the message, to make sure that this
            # information isn't lost if this exception is reraised again
            new_message = ', '.join(str(arg) for arg in new_exc.args)
        else:
            new_message = ""
        new_message += '\n\nOriginal exception:\n\t' + orig_exc_type.__name__
        if hasattr(orig_exc_value, 'args') and len(orig_exc_value.args) > 0:
            if getattr(orig_exc_value, 'reraised', False):
                new_message += ': ' + str(orig_exc_value.args[0])
            else:
                new_message += ': ' + ', '.join(str(arg)
                                                for arg in orig_exc_value.args)
        new_exc.args = (new_message,) + new_exc.args[1:]

    new_exc.__cause__ = orig_exc_value
    new_exc.reraised = True
    raise type(new_exc), new_exc, orig_exc_traceback


def collect_tag(outputs, tag):
    rval = []
    seen = set()

    def _collect_tag(outputs):
        for output in outputs:
            if output in seen:
                continue
            seen.add(output)
            if hasattr(output.tag, tag):
                rval.append(getattr(output.tag, tag))
            owner = output.owner
            if owner is None or owner in seen:
                continue
            seen.add(owner)
            inputs = owner.inputs
            _collect_tag(inputs)
    _collect_tag(outputs)
    return rval

def collect_parameters(outputs):
    rval = []
    seen = set()
    
    def _collect_parameters(otputs):
        for output in outputs:
            if output in seen:
                continue
            seen.add(output)
            owner = output.owner
            if owner is None:
                if isinstance(output, SharedVariable) and getattr(output.tag, 'trainable', False):
                    rval.append(output)
            else:
                if not owner in seen:
                    seen.add(owner)
                    inputs = owner.inputs
                    _collect_parameters(inputs)
    _collect_parameters(outputs)
    return rval


def attach_context(context, inputs):
    ret = []
    for input in inputs:
        reti = input.copy()
        reti.tag.context = context
        ret.append(reti)
    return ret

def find_context(input, location=[]):
    seen = set()
    def _find_context(inputs):
        for input in inputs:
            if hasattr(input.tag, 'context'):
                return input.tag.context
            if input in seen:
                continue
            seen.add(input)
            
            owner = input.owner
            if owner is None or owner in seen:
                continue
            seen.add(owner)
            inputs = owner.inputs
            return _find_context(inputs)
        return None
    
    context = _find_context([input])
    for node in seen: #cache for later use
        node.tag.context = context
    return context.subconf(location)