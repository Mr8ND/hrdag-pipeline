# This functions were taken from Patrick Ball's HRDAG blog post
# https://hrdag.org/tech-notes/clustering-and-solving-the-right-problem.html

import itertools as it 
import numpy as np
import datetime, os
import fastcluster

from collections import defaultdict
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


def make_cdm(hashids, cp, prob_col):
    ''' given a list of hashids, 
        a dataframe with the similarity scores of some of pairs of hashids
        and the dataframe column containing the score values
        return a condensed distance matrix for use in clustering
    '''
    hashids = sorted(list(hashids))
    hashptr = {k: i for i, k in enumerate(hashids)}
    num_ids = len(hashids)
    indexer = make_indexer(hashptr)

    cdm_len = num_ids * (num_ids - 1) / 2 
    cdm = np.ones([cdm_len])  # assume max dissimilarity 
    
    if num_ids < 500:
        # if the cluster is small, look at all the pairs 
        # This is O(n**2), but there's no setup time. So when 
        # n is small, this is good. 
        for hash1, hash2 in it.combinations(hashids, 2): 
            try:
                similarity = cp.loc[(hash1, hash2), prob_col]
            except KeyError: 
                continue 
            index = indexer(hash1, hash2)
            cdm[index] = 1.0 - similarity  # change similarity into distance
    else: 
        # if the cluster is not small, subset and check the known pairs 
        # the cp subset (over the 55M pairs) takes a constant 11 seconds.
        cpsub = cp.loc[(cp.hash1.isin(hashptr)) & (cp.hash2.isin(hashptr))]
        # then iterating over all of the pairs found is O(n) 
        for row in cpsub.itertuples(): 
            index = indexer(row.hash1, row.hash2)
            # change similarity into distance
            cdm[index] = 1.0 - getattr(row, prob_col)
    
    return cdm


def fcluster_one_cc(cc, cp, prob_col, verbose=False):
    def now():
        return '{:%M:%S}'.format(datetime.datetime.now())

    if verbose:
        msg = 'at {} on pid={}: starting HAC on new cluster ({} recs)'
        print(msg.format(now(), os.getpid(), len(cc)))
    
    # handle the simple cases immediately
    if len(cc) <= 2:
        return cc
        
    cdm = make_cdm(cc, cp, prob_col)
    if verbose: 
        msg = 'at {} on pid={}: made CDM, heading into HAC'
        print(msg.format(now(), os.getpid(), len(cc)))

    # from http://danifold.net/fastcluster.html. 
    clustering = fastcluster.linkage(cdm, method='average', preserve_input=False)
    cl = fcluster(clustering, t=0.5, criterion="distance")
    
    clustersd = defaultdict(list)
    for i, mgi in enumerate(cl):
        clustersd[mgi].append(list(cc)[i])

    if verbose:
        msg = '{} on pid={}: finished HAC on new cluster ({} clusters)'
        print(msg.format(now(), os.getpid(), len(clustersd)))

    return [v for v in clustersd.values()]