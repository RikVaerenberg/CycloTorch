import pytest
from scipy.io import loadmat   
from cyclotorch import cpsd
import numpy as np
from scipy.signal import csd

def test_mathworks_example():
    data = loadmat('tests/Matfiles/cspd_mathworks_example.mat')
    fs = 10
    nfft = 1024
    win_length = 500
    n_overlap = 250
    f = data['f']
    cspd_target_mat = data['psxy'].flatten()
    x = data['x'].flatten()
    y = data['y'].flatten()
    cspd_calc = cpsd.cpsd(x,y,win_length,nfft=nfft,n_overlap=n_overlap,fs=fs)
    cspd_target = csd(x,y,fs=fs,nperseg=win_length,noverlap=n_overlap,nfft=nfft,detrend=False,return_onesided=False)[1]
    #Note the complex conjugate with respect to the scipy implementation (following matlab implementation convention)
    assert np.allclose(cspd_target.conj(),cspd_calc.cpu().numpy(),rtol=1e-4,atol=1e-4)
    assert np.allclose(cspd_target_mat,cspd_calc.cpu().numpy(),rtol=5e-3,atol=3e-4)
    
if __name__ == '__main__':
    test_mathworks_example()