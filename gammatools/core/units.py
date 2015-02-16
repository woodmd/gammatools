import numpy as np

class Units(object):

    s = 1.0
    min = 60.*s
    hr = 60.*min
    day = 24.*hr
    week = 7.*day
    year = 365.*day
    
    pc = 3.08568e18   # pc to cm
    kpc = pc*1e3      # kpc to cm
    msun = 1.98892e33 # solar mass to g
    gev = 1.78266e-24 # gev to g    
    mev = 1E-3*gev
    tev = 1E3*gev
    ev = 1E-9*gev

    log10_mev = np.log10(mev)
    log10_gev = np.log10(gev)
    log10_tev = np.log10(tev)

    _mev = 1./mev
    _gev = 1./gev
    _tev = 1./tev

    erg = 1./1.602177E-12*ev
    g = 1.0

    m = 1E2
    km = 1E3*m

    m2 = m**2
    km2 = km**2
    hr = 3600.
    deg = np.pi/180.
    _deg = deg**(-1)
    deg2 = deg**2
    _deg2 = deg2**(-1)

    msun_pc3 = msun*np.power(pc,-3) 
    msun_kpc3 = msun*np.power(kpc,-3)
    msun2_pc5 = np.power(msun,2)*np.power(pc,-5)
    msun2_kpc5 = np.power(msun,2)*np.power(kpc,-5)
    gev2_cm5 = np.power(gev,2)
    gev_cm3 = np.power(gev,1)
    gev_cm2 = np.power(gev,1)
    gev_m2 = np.power(gev,1)/m2
    erg_cm2 = erg
    g_cm3 = 1.0
    cm3_s = 1.0
        
    
    @classmethod
    def parse(cls,s):

        if not isinstance(s,str): return s

        m = re.search('([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)(\s+)?(\w+)?',s)

        if m is None: return s

        v = float(m.group(1))
        units = m.group(4)
        
        if not m.group(4) is None:
            v *= cls.__dict__[units]

        return v

    @classmethod
    def convert(cls,x,from_units,to_units):

        u0 = from_units.lower()
        u1 = to_units.lower()

        u0 = cls.__dict__[u0]
