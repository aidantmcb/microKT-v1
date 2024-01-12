import numpy as np

from astropy.table import Table
from astropy.io import fits


CAresdir = '/uufs/astro.utah.edu/common/home/u1371365/Data/230420_CAResiduals/'
CAMADGICSresdir = '/uufs/astro.utah.edu/common/home/u1371365/Data/230829_MADGICSResiduals/'
respath  = '/uufs/astro.utah.edu/common/home/u1371365/StellarResidualsSpring2022/Residuals/'


def get_ca_res(fname):
    return str(CAresdir + str(fname))

def get_madgics_res(fname):
    return str(CAMADGICSresdir) + str(fname)

meta = Table(fits.open('/uufs/astro.utah.edu/common/home/u1371365/StellarResidualsSpring2022/Residuals/meta.fits')[1].data)

def get_medres(teff, logg, m_h, medres_dir = '/uufs/astro.utah.edu/common/home/u1371365/StellarResidualsSpring2022/Residuals/'):
    rowselect = np.where(np.logical_and.reduce(
                    [teff >= meta['TEFF_MIN'], teff < meta['TEFF_MAX'], 
                    logg >= meta['LOGG_MIN'], logg < meta['LOGG_MAX'],
                   m_h >= meta['M_H_MIN'], m_h < meta['M_H_MAX']]))[0]
    
    row = meta[rowselect]
    filename = row['FNAME'].item()
    return medres_dir + filename
