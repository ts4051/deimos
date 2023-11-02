'''
Tools for data caching

Tom Stuttard
'''

#TODO Needs more documentation

import os, copy, base64, struct, hashlib
from io import IOBase
from collections.abc import Iterable
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle
from pickle import PickleError, PicklingError

def hash_obj(obj, hash_to='int', full_hash=True):
    '''
    A function for hashing arbitrary objects

    Copied from PISA (https://github.com/icecube/pisa/blob/master/pisa/utils/hash.py)
    '''

    if hash_to is None:
        hash_to = 'int'
    hash_to = hash_to.lower()

    pass_on_kw = dict(hash_to=hash_to, full_hash=full_hash)

    # TODO: convert an existing hash to the desired type, if it isn't already
    # in this type
    if hasattr(obj, 'hash') and obj.hash is not None and obj.hash == obj.hash:
        return obj.hash

    # Handle numpy arrays and matrices specially
    if isinstance(obj, (np.ndarray, np.matrix)):
        if full_hash:
            return hash_obj(obj.tostring(), **pass_on_kw)
        len_flat = obj.size
        stride = 1 + (len_flat // FAST_HASH_NDARRAY_ELEMENTS)
        sub_elements = obj.flat[0::stride]
        return hash_obj(sub_elements.tostring(), **pass_on_kw)

    # Handle an open file object as a special case
    if isinstance(obj, IOBase):
        if full_hash:
            return hash_obj(obj.read(), **pass_on_kw)
        return hash_obj(obj.read(FAST_HASH_FILESIZE_BYTES), **pass_on_kw)

    # Convert to string (if not one already) in a fast and generic way: pickle;
    # this creates a binary string, which is fine for sending to hashlib
    if not isinstance(obj, str):
        try:
            pkl = pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)
        except (PickleError, PicklingError, TypeError):
            # Recurse into an iterable that couldn't be pickled
            if isinstance(obj, Iterable):
                return hash_obj([hash_obj(subobj) for subobj in obj],
                                **pass_on_kw)
            else:
                logging.error('Failed to pickle `obj` "%s" of type "%s"',
                              obj, type(obj))
                raise
        obj = pkl

    if full_hash:
        try:
            md5hash = hashlib.md5(obj)
        except TypeError:
            md5hash = hashlib.md5(obj.encode())
    else:
        # Grab just a subset of the string by changing the stride taken in the
        # character array (but if the string is less than
        # FAST_HASH_FILESIZE_BYTES, use a stride length of 1)
        stride = 1 + (len(obj) // FAST_HASH_STR_CHARS)
        try:
            md5hash = hashlib.md5(obj[0::stride])
        except TypeError:
            md5hash = hashlib.md5(obj[0::stride].encode())

    if hash_to in ['i', 'int', 'integer']:
        hash_val, = struct.unpack('<q', md5hash.digest()[:8])
    elif hash_to in ['b', 'bin', 'binary']:
        hash_val = md5hash.digest()[:8]
    elif hash_to in ['h', 'x', 'hex', 'hexadecimal']:
        hash_val = md5hash.hexdigest()[:16]
    elif hash_to in ['b64', 'base64']:
        hash_val = base64.b64encode(md5hash.digest()[:8], '+-')
    else:
        raise ValueError('Unrecognized `hash_to`: "%s"' % (hash_to,))
    return hash_val


class Cachable(object) :
    '''
    Base class that can be used to give member functions caching ability
    '''

    def __init__(self, cache_dir) :

        # Store args
        self.cache_dir = cache_dir

        # Create dir
        if self.cache_dir is not None :
            if not os.path.exists(self.cache_dir) :
                os.mkdir(self.cache_dir)


    def get_state_dict(self) :
        '''
        Get the state dict of the class instance
        '''
        return copy.deepcopy(self.__dict__)


    def get_func_call_hash(self, func_args_dict) :
        '''
        Get the hash of the steerable parameters to a function call
        e.g. the function args and the state of the class instance
        '''
        func_call_dict = { k:v for k,v in func_args_dict.items() if k != "self" }
        func_call_dict.update(self.get_state_dict())
        func_call_hash = hash_obj(func_call_dict)
        return func_call_hash


    def get_cache_file_path(self, func_name, func_call_hash) :
        '''
        Get file path
        '''
        return os.path.join( self.cache_dir, "%s_%s.pckl"%(func_name, func_call_hash) )


    def load_cached_results(self, func_name, func_args_dict) :
        '''
        Load any existing cached results for this function call
        '''

        # Get a hash of all steerable params to this func call (e.g. function inputs and class state)
        func_call_hash = self.get_func_call_hash(func_args_dict)

        # Check if file already exists
        cache_file_path = self.get_cache_file_path(func_name, func_call_hash)
        if os.path.isfile(cache_file_path) :

            # Load the file
            with open(cache_file_path, "rb") as f:
                results = pickle.load(f)

        else :
            # No cached result found
            results = None

        return results, func_call_hash


    def save_results_to_cache(self, func_name, func_call_hash, results) :
        '''
        Save results to a cache file
        '''
        cache_file_path = self.get_cache_file_path(func_name, func_call_hash)
        with open(cache_file_path,"wb") as f :
            pickle.dump( results, f, protocol=-1 ) #protocol=-1 -> use best available (otherwise it defaults to something very slow)

        # print("Saved cache : %s" % cache_file_path)



#
# Test
#

if __name__ == "__main__" :


    #
    # Example class
    #

    class Foo(Cachable) :

        def __init__(self, cache_dir) :
            Cachable.__init__(self, cache_dir=cache_dir)


        def bar(self, a, b) :

            # Load cached results, if available
            results, func_call_hash = self.load_cached_results("bar", locals())

            # Compute if required
            if results is None :

                print("Creating data")
                results = {}
                results["a"] = a
                results["b"] = b

                # Store to cache
                print("Saving to cache")
                self.save_results_to_cache("bar", func_call_hash, results)

            # Return results
            return results["a"], results["b"]


    #
    # Test it
    #

    cache_dir = "./cache_test"
    cache_file = os.path.join(cache_dir, "bar.pckl")

    # Remove any existing cache before starting
    if os.path.exists(cache_file) :
        os.path.remove(cache_file)

    foo = Foo(cache_dir=cache_dir)

    print("\nFirst call to `foo.bar(1, 2)` : Should generate data and save a cache")
    print( foo.bar(1,2) )


    print("\nSecond call to `foo.bar(1, 2)` : Should load the cache")
    print( foo.bar(1,2) )


    print("\nFirst call to `foo.bar(1, 3)` (e.g. new args) : Should generate data and save a cache")
    print( foo.bar(1,3) )

    print("\nThird call to `foo.bar(1, 2)` : Should load the cache")
    print( foo.bar(1,2) )

