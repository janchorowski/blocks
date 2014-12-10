'''
Hierarchical configuration module.

Key concepts: the configuration behaves like a dict from `paths` to values. A path is
a string of Python identifiers separated by single forward slashes ('/').

Internally the configuration is a list of `selector`-`value` pairs. A `path` can match a `selector` 
with a given specificity. The specificity us used to reslove ties between multiple selectors matching 
the same path.

Selectors are sequences of identifiers or regular expressions seperated by 
single or double slashes (a double slash matches zero or more levels of the hierarchy).


Missing stuff:
prevent keys from matching prefixes of other keys

Created on Oct 8, 2014

@author: Jan Chorowski
'''

__all__ = ['Conf']

import re
from pprint import pformat

class AmbiguousSpecificity(Exception):
    """
    Exception returned when two or more keys match a path with the same (maximal) specificity.
    """

class Entry(object):
    __slots__ = ['specificity', 'value']
    _base_specificity = (0,0)
    
    def __init__(self, value, specificity=_base_specificity):
        self.specificity = specificity
        self.value = value
    
#useful regexpes
__first_single_slash = re.compile('^/[^/]')
__last_single_slash = re.compile('[^/]/$')
__valid_path = '^[^\d\W]\w*(?:/[^\d\W]\w*)*$' #single / separated python identifier

def check_valid_selectors(selector_parts):
    """
    Selector parts form a selector by logically joining them with '/'.
    
    Validate selector parts. Raise on wrong ones. Selectors have to be:
    - nonempty
    - can't begin or end with a single /
    - can contain // to indicate a wildard match at this level and descendants
    - can not contain more than 2 consecutive slashes 
    """
    for sp in selector_parts:
        if not sp:
            raise Exception("Selector elements can not be null.")
        if re.match(__first_single_slash, sp):
            raise Exception("Selector elements can't begin with a single /. Use // to indicate all descendants, or no / at all")
        if re.match(__last_single_slash, sp):
            raise Exception("Selector elements can't end with a /.")
        if sp.find('///') > -1:
            raise Exception("Selector elements can't contain more than two consecutive /.")
    return True

def check_valid_paths(path_parts):
    """
    Path parts form a path by logically joining them with '/'.
    
    Validate paths. returns True on success, raises on error.
    
    Paths are python identifiers separated by single slashes.
    """
    for pp in path_parts:
        if not pp:
            raise Exception("Path parts can not be null.")
        if not re.match(__valid_path, pp):
            raise Exception('Path parts must be formed as python indentifiers separated by single slashes.')
    return True

def _selector_path_join_valid(*key_or_path_parts):
    """
    Internal method.
    
    Join parts assuming they are valid making sure to not introduce extra slashes when concatenating // wildcards.
    """
    #now we know that path components do not begin or end with single slashes
    joined = '/'.join(key_or_path_parts)
    #we could have introduced blocks of two or more slashes (worst case - 5 slashes from joining two wildcards
    return re.sub('//+','//', joined)

def _split_to_parts_valid(path_or_segments):
    """
    Split path segments into parts assuming they were valid.
    """
    #note: this assumes that there are no empty path parts
    temp_parts = sum((p.split('/') for p in path_or_segments), [])
    parts = ['']
    for p in temp_parts:
        if p:
            parts.append(p)
        else:
            if parts[-1]!='//':
                parts.append('//')
    return parts[1:]

def _match(selector_parts, path_parts, specificity, do_prefix_match):
    """
    Internal function: little validation is made.
    
    Match a path (split into parts) to a key (path selector with wildcards). 
    Return None if no match was found or the specificity of the best match.
    
    Specificity measures how specific a match was. Its computation is similar to CSS:
    - the score is a tuple (count of exact name matches, count of regexp name patches)
    - the tuples are compared lexicographically
    - a single wildcard match // is the weakest match possible
    - we forbid matches with the same score
    
    Arguments:
    ==========
    
    - selector_parts: (list of str) selector split into parts (wildcards '//', and regexpes)
    - path_parts: (list of str) path split into single indentifiers along it
    - specificity: base specificity of the match. For recurrent calls only.
    - do_prefix_match: if True only checks if the path matches the beginning of a selector. 
      It returns the unmatched part of the selector.
    
    
    """
    if path_parts == []:
        if do_prefix_match:
            return selector_parts, specificity
        
        if selector_parts==[]:
            return specificity
        else:
            return None
    else:
        if selector_parts == []:
            return None
    
    selector_head = selector_parts[0]
    selector_tail = selector_parts[1:]
    if selector_head=='//':
        #the // matches are minimal: they match only as little as possible to match the next path component
        assert selector_tail #Conf.__setitem__ ensures that no selector can end in a wildcard
        sekector_tail_head =selector_tail[0]
        for skipped_levels in xrange(len(path_parts)): #allow the match to th empty path
            skipped_head = path_parts[skipped_levels]
            is_match = re.match(sekector_tail_head, skipped_head)
            if is_match and is_match.end()==len(skipped_head):
                return _match(selector_tail, path_parts[skipped_levels:], 
                              specificity, do_prefix_match)
        
        #if we are here, then no path component matched the selector after wildcard
        if do_prefix_match:
            return selector_parts, specificity
        else:
            return None
    
    path_head = path_parts[0]
    is_match = re.match(selector_head, path_head)
    if is_match and is_match.end()==len(path_head):
        new_specificity=list(specificity)
        if selector_head==path_head:
            new_specificity[0] += 1
        else:
            new_specificity[1] += 1
        return _match(selector_tail, path_parts[1:],
                       new_specificity, do_prefix_match)
    else:
        return None
    
class Conf(object):
    """
    The basic configuration object: a dict from path selectors to values
    
    Path selectors are similar to XPath:
    Selectors are composed of:
    - regular expressions for names
    - path separators /
    - an empty name (//) allows for a minimal match over this and lower levels
    - a path selector cannot end with a wildcard (//). Exception: the value must be a Conf object to recourse into. 
    """

    _special_selectors = {'name': lambda self: self._location[-1],
                          'location': lambda self: _selector_path_join_valid(*self._location)}
    
    def __init__(self, conf_dict={}, location=[], _reuse_conf_dict=False):
        if isinstance(location, str):
            location = [location]
        check_valid_paths(location)
        self._location=_split_to_parts_valid(location)
        
        if _reuse_conf_dict:
            self._conf_dict = conf_dict
        else:
            self._conf_dict = {}
            self.update(conf_dict)
    
    @staticmethod
    def _match(selector, path_parts, specificity, do_prefix_match=False):
        check_valid_selectors(selector) #shouldn't be needed if we assume no one messes with our dict
        selector_parts = _split_to_parts_valid([selector])
        return _match(selector_parts, path_parts, 
                      specificity=specificity,
                      do_prefix_match=do_prefix_match)
    
    def __str__(self):    
        return 'Conf(%s)' % ( str(dict(self._iter_location_restricted_entries())),)
    
    def __repr__(self):
        return 'Conf(%s)' % ( pformat(dict(self._iter_location_restricted_entries())), )
    
    def _debugprint(self):
        ret = []
        for selector,entry in self._conf_dict.iteritems():
            sel_suffix = self._match(selector, self._location, entry.specificity, do_prefix_match=True)
            if sel_suffix is not None:
                ret.append('%s:%s -> loc + %s' % (selector, entry.value, _selector_path_join_valid(*sel_suffix)))
            else:
                ret.append('%s:%s -> None' % (selector, entry.value))
        ret = ['_location: %s' %(self._location, )] + sorted(ret)
        return '\n'.join(ret)
    
    def _get_path_parts(self, path):
        _path = list(self._location)
        _path.extend(path)
        check_valid_paths(_path)
        return _split_to_parts_valid(_path)
    
    def _get_selector_parts(self, selector):
        _selector = list(self._location)
        _selector.extend(selector)
        check_valid_selectors(_selector)
        return _split_to_parts_valid(_selector)
    
    def __getitem__(self, path):
        if isinstance(path, str):
            path = (path, )
        path_parts = self._get_path_parts(path)
        
        #special cases     
        if path_parts[-1] in self._special_selectors:
            if len(path_parts)==1+len(self._location):
                return self._special_selectors[path_parts[-1]](self)
            else:
                return self.subconf(*path_parts[:-1])[path_parts[-1]]
        
        matches = []
        for selector,entry in self._conf_dict.iteritems():
            #do we need to validate the key here?
            match_specificity = self._match(selector, path_parts, Entry._base_specificity) #entry.specificity)
            if match_specificity:
                matches.append((match_specificity, selector, entry)) # entry.value))
        if not matches:
            raise KeyError(_selector_path_join_valid(*path_parts))
        if len(matches)==1:
            return matches[0][-1]
        #multiple latches - sort by specificity to resolve ambiguities
        matches.sort(reverse=True) #best match is first
        if matches[0][0] > matches[1][0]:
            return matches[0][-1] #ok, the best match is more specific than any other
        raise AmbiguousSpecificity('Selectors %s match path %s with the same specificity of %s' %(
                ' and '.join(m[1] for m in matches if m[0]==matches[0][0]),
                _selector_path_join_valid(*path_parts), matches[0][0]))

    def get(self, path, default=None):
        try:
            return self.__getitem__(path)
        except KeyError:
            return default

    def _iter_location_restricted_entries(self):
        for selector,entry in self._conf_dict.iteritems():
            sel_suffix, sel_specificity = self._match(selector, self._location, 
                                                      Entry._base_specificity, do_prefix_match=True)
            #sel_specificity[0]-=len(self._location)
            if sel_suffix is not None:
                yield _selector_path_join_valid(*sel_suffix), entry#, Entry(TODO)

    def __setitem__(self, selector, value):
        if isinstance(selector, str):
            selector = (selector, )
        selector_parts = self._get_selector_parts(selector)
        s = _selector_path_join_valid(*selector_parts)
        
        if selector_parts[-1] in self._special_selectors:
            raise Exception('Special selector %s can not be set manually!' %(s[-1], ))
        
        if not isinstance(value, Conf):
            if s.endswith('//'):
                raise Exception("Selectors for values can not end with a wildcard ('//')")
            self._conf_dict[s] = value
        else:
            #v's dict is already flattened: just do the concatenation
            
            #currently we don't support triming of a configuration to a 
            #locatiot
            for sub_selector, sub_value in value._iter_location_restricted_entries():
                self[s, sub_selector] = sub_value

    def subconf(self, *name):
        if isinstance(name, str):
            name = (name, )
        name_parts = self._get_path_parts(name)
        ret = Conf(self._conf_dict, name_parts, _reuse_conf_dict=True)
        return ret

    def update(self, opts):
        for s,v in opts.iteritems():
            check_valid_selectors(s)
            self[s] = v
        return self
