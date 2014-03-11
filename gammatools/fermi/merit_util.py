from gammatools.core.histogram import *

def createBkgRateHist(dset,var_list,var_axes,cuts=''):
        
    h = HistogramND.createFromTree(dset.chain(),var_list,var_axes,cuts)

    livetime = dset.getLivetime()    
    ewidth = h.axis(0).width()[[slice(None)] + (len(var_list)-1)*[None]]
    egy = 10**(h.axis(0).center())[[slice(None)] + (len(var_list)-1)*[None]]
    
    h *= egy/(livetime*ewidth*np.log(10.)*4*np.pi*dset.config['fraction'])
    return h

def createAcceptanceHist(dset,var_list,var_axes,mc_eaxis,mc_cthaxis=None,
                         cuts='',mc_evar='McLogEnergy'):
    
    """Compute acceptance [m^2 sr] of a MC gamma-ray data set.
    Generates an n-dimensional histogram of acceptance versus log
    of the true gamma-ray energy, cosine of the true inclination angle
    (optional), and one or more reconstruction parameters."""

#        mc_cthrange['label'] = 'Cos $\\theta$'
#        mc_erange['label'] = 'Energy [log$_{10}$(E/MeV)]'

    if var_list is None: var_list = []
    if var_axes is None: var_axes = []
    
    egy_bins = mc_eaxis.edges()
    cth_bins = None

    slices = len(var_list)*[None]
    
    if not mc_cthaxis is None:
        mc_var_list = [mc_evar,'-McZDir']
        mc_var_axes = [mc_eaxis,mc_cthaxis]
        cth_bins = mc_cthaxis.edges()
    else:
        mc_var_list = [mc_evar]
        mc_var_axes = [mc_eaxis]
            
    var_list = mc_var_list + var_list 
    var_axes = mc_var_axes + var_axes
            
    h = HistogramND.createFromTree(dset.chain(),var_list,var_axes,cuts)
        
    thrown_events = dset.getThrownEvents(egy_bins,cth_bins)
    
    if mc_cthaxis:

        domega = 2*np.pi*mc_cthaxis.width()[[None,slice(None)] + slices]
        eff = h.counts()/thrown_events[[slice(None),slice(None)] + slices]
        eff_var = eff*(1-eff)/thrown_events[[slice(None),slice(None)] + slices]
        acc = eff*6.0/dset.config['fraction']*domega
        acc_var = eff_var*(6.0/dset.config['fraction']*domega)**2
    else:
        eff = h.counts()/thrown_events[[slice(None)] + slices]
        eff_var = eff*(1-eff)/thrown_events[[slice(None)] + slices]
        acc = eff*6.0*(2*np.pi)/dset.config['fraction']
        acc_var = eff_var*(6.0*(2*np.pi)/dset.config['fraction'])**2
        
    h._counts = acc
    h._var = acc_var

    return h
