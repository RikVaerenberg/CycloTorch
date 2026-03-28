import pytest
from scipy.io import loadmat   
from cyclotorch import CSC_estimators 
import numpy as np




def test_fast_sc():
    matfile = r"tests/Matfiles/jeromefastsc_example.mat"
    mat = loadmat(matfile)
    signal = mat['x'][:,0]
    S_jerome = mat['S'].T
    alpha = mat['alpha'].flatten()
    f_j = mat['f'].flatten()

    window_len = 64

    segment_max = 0.09
    alpha_arr,f,scd = CSC_estimators.Fast_SC(
            signal,
            alpha_max=0.1,
            fs=1.0,
            window_len=128,
            convention='asymmetric_positive'
        )
    f = f[f>=0].numpy()


    scd_segment = scd[alpha_arr<segment_max,:][:,:len(f)].numpy()
    jerome_segment = S_jerome[alpha<segment_max,:][:,f_j>=0]
    
    mask = (f[np.newaxis, :] + alpha[alpha<segment_max, np.newaxis]/2) > 0.5
    scd_segment[mask] =0
    jerome_segment[mask] = 0
    
    
    assert np.allclose(scd_segment,jerome_segment,rtol=1e-4,atol=0.03)

