'''
Hierarchical configuration module

Selectors are specified keys of the configuration dict.
Paths are fully specified names of objects that are fetched form the configuration.

Paths are python identifiers separated by single slashes

Selectors are sequences of identifiers or regular expressions seperated by 
single or double slashes (a double slash matches zero or more levels of the hierarchy).


Missing stuff:
prevent keys from matching prefixes of other keys

Created on Oct 8, 2014

@author: Jan
'''

__all__ = ['Conf']

import re
from pprint import pprint

class AmbiguousSpecificity(Exception):
    pass

#useful regexpes
__first_single_slash = re.compile('^/[^/]')
__last_single_slash = re.compile('[^/]/$')
__valid_path = '^[^\d\W]\w*(?:/[^\d\W]\w*)*$' #single / separated python identifier

def check_valid_selectors(selector_parts):
    """
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
    Validate paths. returns True on success, raises on error.
    
    Paths are python identifiers separated by single slashes.
    """
    for pp in path_parts:
        if not pp:
            raise Exception("Path elements can not be null.")
        if not re.match(__valid_path, pp):
            raise Exception('Path elements must be formed as python indentifiers separated by single slashes.')
    return True

def _selector_path_join_valid(*key_or_path_parts):
    """
    Validate components then join making sure to not introduce extra slashes when concatenating // wildcards.
    """
    #now we know that path components do not begin or end with single slashes
    joined = '/'.join(key_or_path_parts)
    #we could have introduced blocks of two or more slashes (worst case - 5 slashes from joining two wildcards
    return re.sub('//+','//', joined)

def _split_to_parts_valid(path_or_segments):
    """
    Split path segments into parts.
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

_base_specificity = [0,0]
def _match(selector_parts, path_parts, allow_prefix_match, specificity):
    """
    Match a path (split into parts) to a key (path selector with wildcards). Return None if no match was found or the specificity of the best match.
    
    Specificity measures how specific a match was. Its computation is similar to CSS:
    - the score is a tuple (count of exact name matches, count of regexp name patches)
    - the tuples are compared lexicographically
    - a single wildcard match // is the weakest match possible
    - we forbid matches with the same score
    """
    if path_parts == []:
        if allow_prefix_match or selector_parts==[]:
            return specificity
        else:
            return None
    else:
        if selector_parts == []:
            return None
    
    selector_head = selector_parts[0]
    selector_tail = selector_parts[1:]
    if selector_head=='//':
        matches = [None] # no match by default
        for skipped_levels in xrange(len(path_parts)+1): #allow the match to th empty path
            matches.append( _match(selector_tail, path_parts[skipped_levels:], 
                                   allow_prefix_match, specificity) ) 
        return max(matches)
    
    path_head = path_parts[0]
    is_match = re.match(selector_head, path_head)
    if is_match and is_match.end()==len(path_head):
        new_specificity=list(specificity)
        if selector_head==path_head:
            new_specificity[0] += 1
        else:
            new_specificity[1] += 1
        return _match(selector_tail, path_parts[1:],
                      allow_prefix_match, new_specificity)
    else:
        return None
    
class Conf(object):
    """
    The basic configuration object: a dict from paths to values
    
    Path specification is similar to XPath:
    Paths are composed of:
    - regular expressions for names
    - path separators /
    - an empty name (//) means descend into this and all levels below 
    """
    
    def __init__(self, conf_dict={}, location=[], _reuse_conf_dict=False):
        if isinstance(location, str):
            location = [location]
        check_valid_paths(location)
        self._location=_split_to_parts_valid(location)
        
        if _reuse_conf_dict:
            self._cd = conf_dict
        else:
            self._cd = {} #short for conf dict
            #flatten the namespace
            for s,v in conf_dict.iteritems():
                check_valid_selectors(s)
                if isinstance(v, Conf):
                    #v's dict is already flattened: just do the concatenation
                    
                    #currently we don't support triming of a configuration to a 
                    #locatiot
                    assert v._location == []
                    
                    for ss, sv in v._cd.iteritems():
                        assert not isinstance(sv, Conf)
                        self._cd[_selector_path_join_valid(s,ss)] = sv
                else:
                    self._cd[s] = v
    
    @staticmethod
    def _match(selector, path_parts, allow_prefix_match=False):
        check_valid_selectors(selector) #shouldn't be needed if we assume no one messes with out dict
        selector_parts = _split_to_parts_valid([selector])
        return _match(selector_parts, path_parts, 
                      allow_prefix_match=allow_prefix_match, 
                      specificity=_base_specificity)
    
    def _get_location_restricted_cd(self):
        ret_cd = {}
        for s,v in self._cd.iteritems():
            if self._match(s, self._location, allow_prefix_match=True):
                ret_cd[s]=v
        return ret_cd
    
    def __str__(self):    
        return str(self._get_location_restricted_cd())
    
    def __repr__(self):
        return repr(self._get_location_restricted_cd())
    
    def pprint(self):
        return pprint(self._get_location_restricted_cd())
    
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
        matches = []
        for k,v in self._cd.iteritems():
            #do we need to validate the key here?
            match_specificity = self._match(k, path_parts)
            if match_specificity:
                matches.append((match_specificity, k, v))
        if not matches:
            raise KeyError(_selector_path_join_valid(*path_parts))
        if len(matches)==1:
            return matches[0][-1]
        #multiple latches - sort by specificity to resolve ambiguities
        matches.sort(reverse=True) #best match is first
        if matches[0][0] > matches[1][0]:
            return matches[0][-1] #ok, the best match is more specific than any other
        raise AmbiguousSpecificity('Selectors %s match path %s with the same specificity of %s' %(
                ' and '.join(m[0] for m in matches if m[0]==matches[0][0]),
                _selector_path_join_valid(*path_parts), matches[0][0]))

    def get(self, path, default=None):
        try:
            return self.__getitem__(path)
        except KeyError:
            return default

    def __setitem__(self, selector, value):
        if isinstance(selector, str):
            path = (selector, )
        path_parts = self._get_selector_parts(path)
        s = _selector_path_join_valid(*path_parts)
        self._cd[s]=value

    def subconf(self, name):
        check_valid_paths([name])
        ret = Conf(self._cd, self._location + [name], _reuse_conf_dict=True)
        ret['name'] = name
        return ret

    def update(self, opts):
        for s,v in opts:
            check_valid_selectors(s)
            assert not isinstance(v, Conf)
            self[s] = v
        return self
