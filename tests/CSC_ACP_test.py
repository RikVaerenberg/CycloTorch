import pytest
from scipy.io import loadmat   
from cyclotorch.CSC_estimators import CSC_ACP
import numpy as np


def test_mathworks_example():
    data = loadmat('tests/Matfiles/scd_acp_mathworks_example.mat')
    fs = 1
    nfft = data['nfft'].flatten().item()
    win_length = data['Nw'].flatten().item()
    n_overlap =  data['Nv'].flatten().item()
    x = data['x'].flatten()
    f = data['f']
    alpha_arr = data['alpha'].flatten()
    scd_mat = data['S'].T
    coh_mat = data['C'].T
    scd_tc = CSC_ACP(x,x,alpha_arr,win_length,nfft=nfft,n_overlap=n_overlap,fs=fs,convention='symmetric')
    coh_tc = CSC_ACP(x,x,alpha_arr,win_length,nfft=nfft,n_overlap=n_overlap,fs=fs,convention='symmetric',coherence=True)
    
    assert np.allclose(scd_mat,scd_tc.cpu().numpy(),rtol=1e-4,atol=2e-4)
    assert np.allclose(coh_mat,coh_tc.cpu().numpy(),rtol=1e-4,atol=5.5e-4)


    
if __name__ == '__main__':
    test_mathworks_example()