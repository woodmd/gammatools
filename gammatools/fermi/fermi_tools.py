import numpy as np

def GetCountsMap(binnedAnalysis):
    """ Get the shape of the observed counts map
    from a BinnedAnalysis object

    binnedAnalysis:    The BinnedAnalysis object 

    returns  np.ndarray( (nEBins, nPixX, nPixY), 'd' )
    """
    ll = binnedAnalysis.logLike    
    shape = GetCountsMapShape(ll.countsMap())
    a = np.ndarray( (shape[2],shape[1],shape[0]), 'f' )
    a.flat = binnedAnalysis.logLike.countsMap().data()
    return a


def GetCountsMapShape(countsMap):
    """ Get the shape of the observed counts map
    from a CountsMap object

    countsMap:    The CountsMap object     

    returns tuple ( nEBins, nPixX, nPixY )
    """
    n0 = countsMap.imageDimension(0)
    n1 = countsMap.imageDimension(1)
    try:
        n2 = countsMap.imageDimension(2)
    except:
        return (n0,n1)
    return (n0,n1,n2)
