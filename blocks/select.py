"""
The select module offers means to select sets of bricks and to filter
lists of variables, application calls and the like by their attributes,
brick ownership etc.

Selecting bricks
----------------

Bricks are selected using expressions similar to XPath formed of:
- the child selector '/'
- the all children selector '//'
- the name matcher (a valid python identifier)

Guys comment do we want this:
- the regexp name matcher in square brackets

- the attribute existence matcher @attribute
- the attribute value matcher @attribute==value
"""

import logging
import re

import warnings

logger = logging.getLogger(__name__)


class SelectorParseException(Exception):
    """
    Exception raised when the selector expression can't be parsed.
    """


class AbstractSelector(object):
    @staticmethod
    def partial_parse_path(path):
        """
        Check whether the be begining of the path can be interpreted
        as this selector. If it can, return the number of characters
        this type of selector matches and the selector.
        If it cannot, return 0, None.
        """
        return 0, None

    def _match(self, brick):
        "Helper for enumerate, do not use directly"
        return False

    def enumerate(self, bricks):
        for brick in bricks:
            if self._match(brick):
                yield brick

    def _enumerate_paths(self, paths_to_bricks):
        for path, brick in paths_to_bricks.iteritems():
            if self._match(brick):
                yield path, brick

    def enumerate_paths(self, bricks, path=''):
        if isinstance(bricks, dict):
            return self._enumerate_paths(bricks)
        else:
            paths_to_bricks = {"{}{}".format(path, b.name): b for b in bricks}
            return self._enumerate_paths(paths_to_bricks)


class ChildSelector(AbstractSelector):
    """
    The single '/' selector - enumerate children

    Parses:
    /
    """

    CHILDREN_FIELD = 'children'

    @staticmethod
    def partial_parse_path(path):
        if path.startswith('/'):
            if path.startswith('//'):
                warnings.warn("The AllDescendantsSelector (//) selector should"
                              "be matched before the Child (/) selector")
                return 0, None
            return 1, ChildSelector()
        else:
            return 0, None

    def __init__(self, **kwargs):
        super(ChildSelector, self).__init__(**kwargs)

    def enumerate(self, bricks):
        for brick in bricks:
            for c in getattr(brick, ChildSelector.CHILDREN_FIELD, []):
                yield c

    def _enumerate_paths(self, paths_to_bricks):
        for path, brick in paths_to_bricks.iteritems():
            for c in getattr(brick, ChildSelector.CHILDREN_FIELD, []):
                yield ("{}/{}".format(path, c.name), c)


class FieldSelector(AbstractSelector):
    """
    The single '.field' selector - enumerate contents of .field

    Parses:
    .field
    """

    FIELD_IDENTIFIER_RE = re.compile('^\.([^\d\W]\w*)')

    @staticmethod
    def partial_parse_path(path):
        m = FieldSelector.FIELD_IDENTIFIER_RE.match(path)
        if m:
            return m.end(), FieldSelector(m.group(1))
        return 0, None

    def __init__(self, field_name, **kwargs):
        super(FieldSelector, self).__init__(**kwargs)
        self.field_name = field_name

    def enumerate(self, bricks):
        for brick in bricks:
            for c in getattr(brick, self.field_name, []):
                yield c

    def _enumerate_paths(self, paths_to_bricks):
        for path, brick in paths_to_bricks.iteritems():
            for c in getattr(brick, self.field_name, []):
                yield ('{}.{}[{}]'.format(path,
                                              self.field_name,
                                              c.name),
                       c)


class AllDescendantsSelector(AbstractSelector):
    """
    The double '//' selector yields a brick and all its descendants.

    Parses:
    //
    """

    @staticmethod
    def partial_parse_path(path):
        if path.startswith('//'):
            return 2, AllDescendantsSelector()
        else:
            return 0, None

    def __init__(self, **kwargs):
        super(AllDescendantsSelector, self).__init__(**kwargs)

    def enumerate(self, bricks):
        def list_subbricks(brick):
            yield brick
            for child in brick.children:
                for c in list_subbricks(child):
                    yield c

        for brick in bricks:
            for r in list_subbricks(brick):
                yield r

    def _enumerate_paths(self, paths_to_bricks):
        def list_subbricks(path, brick):
            yield path, brick
            for child in brick.children:
                for c in  list_subbricks("{}/{}".format(path, child.name),
                                         child):
                    yield c

        for path, brick in paths_to_bricks.iteritems():
            for r in list_subbricks(path, brick):
                yield r


class RegexpSelector(AbstractSelector):
    """
    The [name_regexp] selector matches names matching exactly the given
    regexp.

    Parses:
    -------
    [regular_expression]
    """
    @staticmethod
    def partial_parse_path(path):
        if not path.startswith('['):
            return 0, None
        num_paren = 1
        for offset, char in enumerate(path[1:]):
            if char == '[':
                num_paren += 1
            elif char == ']':
                num_paren -= 1
                if num_paren == 0:
                    break
        pattern_end = offset + 1
        if path[pattern_end] != ']':
            raise SelectorParseException("couldn't parse regexp selector "
                                         "starting with: {}".format(path))
        return pattern_end + 1, RegexpSelector(path[1:pattern_end])

    def __init__(self, name_re, **kwargs):
        super(RegexpSelector, self).__init__(**kwargs)
        self.name_re = re.compile('^{}$'.format(name_re))

    def _match(self, brick):
        return self.name_re.match(brick.name) is not None


class AttributeMatchSelector(AbstractSelector):
    """
    Realize matches by tests on attributes (by default on the name).

    Parses:
    name
    \@attr
    \@attr == value
    """
    PYTHON_IDENTIFIER_RE = re.compile('^[^\d\W]\w*')
    ATTR_CHECK_RE = re.compile('^@([^\d\W]\w*)(\s*==\s*([^\d\W]\w*))?')

    @staticmethod
    def partial_parse_path(path):
        # 1. try a direct match on name
        m = AttributeMatchSelector.PYTHON_IDENTIFIER_RE.match(path)
        if m:
            return m.end(), AttributeMatchSelector('name', m.group(0))
        # 2. try a match on attribute value
        m = AttributeMatchSelector.ATTR_CHECK_RE.match(path)
        if m:
            attr_name, eq_check, attr_val = m.groups()
            return m.end(), AttributeMatchSelector(attr_name, attr_val,
                                           hasattr_test=eq_check is None)
        return 0, None

    def __init__(self, attr_name, attr_val, hasattr_test=False, **kwargs):
        super(AttributeMatchSelector, self).__init__(**kwargs)
        self.attr_name = attr_name
        self.attr_val = attr_val
        self.hasattr_test = hasattr_test

    def _match(self, brick):
        if self.hasattr_test:
            return hasattr(brick, self.attr_name)
        else:
            return (hasattr(brick, self.attr_name) and
                    getattr(brick, self.attr_name) == self.attr_val)


class TerminatorSelector(AbstractSelector):
    """
    A dummy selector class which raises whtn the path is non-empty.
    It is used as a canary at the end of the Selector.ParseOrder list.
    """

    @staticmethod
    def partial_parse_path(path):
        raise SelectorParseException()


class Selector(AbstractSelector):
    ParseOrder = [AllDescendantsSelector, ChildSelector,
                  FieldSelector,
                  RegexpSelector,
                  AttributeMatchSelector,
                  TerminatorSelector]

    def __init__(self, path_or_selector_list, **kwargs):
        super(Selector, self).__init__(**kwargs)
        self.selectors = []
        if not isinstance(path_or_selector_list, list):
            selector_list = [path_or_selector_list]
        for path_or_sel in selector_list:
            if isinstance(path_or_sel, AbstractSelector):
                self.selectors.append(path_or_sel)
            else:
                assert isinstance(path_or_sel, str)
                path = path_or_sel
                while path:
                    for selector_klass in Selector.ParseOrder:
                        offset, sel = selector_klass.partial_parse_path(path)
                        if offset:
                            path = path[offset:]
                            self.selectors.append(sel)
                            break

    def enumerate(self, bricks):
        for sel in self.selectors:
            bricks = set(sel.enumerate(bricks))
        return bricks

    def _enumerate_paths(self, paths_to_bricks):
        for sel in self.selectors:
            paths_to_bricks = dict(sel.enumerate_paths(paths_to_bricks,
                                                        None))
        return paths_to_bricks
