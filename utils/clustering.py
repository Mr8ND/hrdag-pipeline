# This functions were taken from Patrick Ball's HRDAG blog post
# https://hrdag.org/tech-notes/clustering-and-solving-the-right-problem.html

from collections import defaultdict, OrderedDict, Counter
from pprint import pprint
import itertools as it 
import time
import os 
import datetime 
import warnings

import networkx as nx
import pandas as pd
import numpy as np
import fastcluster 
from scipy.cluster.hierarchy import fcluster 

def make_indexer(hashptr):
    ''' the returned indexer function takes two hashes
        and returns the cdm index location. 
        
        the hashptr dict is a dict of all the hashids into their 
        sorted sequence position. 
        
        note that the closure binds the hashptr dict and the dlen
        calculation into the function's enclosing scope. this is a way
        to bind data to the function, very much like a class. 
        
        the index calculation is explained in fgregg's answer to this stackoverflow question:
        https://stackoverflow.com/questions/5323818/condensed-matrix-function-to-find-pairs   
    '''
    d = len(hashptr)
    dlen = d * (d - 1) / 2
    def d_indexer(k1, k2):
        i, j = hashptr[k1], hashptr[k2]
        offset = (d - i) * (d - i - 1) / 2
        index = dlen - offset + j - i - 1
        # index *is* an int, but some of the calcs above may 
        # make it a float which triggers a DeprecationWarning
        # when it's used to index an array (which is the point)
        return int(index) 
    return d_indexer