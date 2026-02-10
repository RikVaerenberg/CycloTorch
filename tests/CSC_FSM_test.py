import pytest
from scipy.io import loadmat   
from cyclotorch.CSC_estimators import CSC_ACP,CSC_FSM
import numpy as np




def test_mathworks_example_FSM():
    data = loadmat('tests/Matfiles/scd_acp_mathworks_example.mat')
    fs = 1
    nfft = data['nfft'].flatten().item()
    win_length = data['Nw'].flatten().item()
    n_overlap =  data['Nv'].flatten().item()
    x = data['x'].flatten()

    f = data['f'].flatten()
    alpha_arr = data['alpha'].flatten()
    scd_mat = data['S'].T
    coh_mat = data['C'].T

    scd_tc = CSC_ACP(x,x,alpha_arr,win_length,nfft=nfft,n_overlap=n_overlap,fs=fs,convention='symmetric')
    coh_tc = CSC_ACP(x,x,alpha_arr,win_length,nfft=nfft,n_overlap=n_overlap,fs=fs,convention='symmetric',coherence=True)
    smooth_len = int(len(x)/nfft*4)


    scd_tc_fsm = CSC_FSM(x,x,alpha_arr,smooth_len=smooth_len,fs=fs,convention='symmetric')    
    coh_tc_fsm  = CSC_FSM(x,x,alpha_arr,smooth_len=smooth_len,fs=fs,convention='symmetric',coherence=True)
    f_tc = np.arange(0,fs,fs/nfft)
    f_fsm = np.arange(0,fs,fs/len(x))
    closest_indices = np.searchsorted(f_fsm, f_tc,sorter=np.argsort(f_fsm))
    
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 6))
    # plt.imshow(np.abs(coh_tc.T.cpu().numpy()), aspect='auto', extent=[alpha_arr.min(), alpha_arr.max(), f_tc[0], f_tc[-1]], origin='lower')
    # plt.colorbar(label='Magnitude')
    # plt.xlabel('Cyclic Frequency (alpha)')
    # plt.ylabel('Frequency (Hz)')
    # plt.title('SCD_TC Magnitude')
    # plt.show(block=False)
    # plt.figure(figsize=(10, 6))
    # plt.imshow(np.abs(coh_mat.T), aspect='auto', extent=[alpha_arr.min(), alpha_arr.max(), f_tc[0], f_tc[-1]], origin='lower')
    # plt.colorbar(label='Magnitude')
    # plt.xlabel('Cyclic Frequency (alpha)')
    # plt.ylabel('Frequency (Hz)')
    # plt.title('SCD mat Magnitude')
    # plt.show(block=False)
    
    # plt.figure(figsize=(10, 6))

    # plt.imshow(np.abs(coh_tc_fsm[:,closest_indices]).T, aspect='auto', extent=[alpha_arr.min(), alpha_arr.max(), f[0], f[-1]], origin='lower')
    # plt.colorbar(label='Magnitude')
    # plt.xlabel('Cyclic Frequency (alpha)')
    # plt.ylabel('Frequency (Hz)')
    # plt.title('SCD fsm Magnitude')
    # plt.show(block=False)
    
    # plt.figure(figsize=(10, 6))
    # plt.imshow(np.abs(scd_tc_fsm[:,closest_indices]-scd_tc).T, aspect='auto', extent=[alpha_arr.min(), alpha_arr.max(), f[0], f[-1]], origin='lower')
    # plt.colorbar(label='Magnitude')
    # plt.xlabel('Cyclic Frequency (alpha)')
    # plt.ylabel('Frequency (Hz)')
    # plt.title('SCD fsm Magnitude')
    # plt.show(block=False)
    
    
    assert np.allclose(scd_tc.cpu().numpy(),scd_tc_fsm[:,closest_indices].cpu().numpy(),rtol=1e-4,atol=0.03)
    assert np.allclose(coh_tc.cpu().numpy(),coh_tc_fsm[:,closest_indices].cpu().numpy(),rtol=1e-4,atol=0.05)


    
if __name__ == '__main__':
    test_mathworks_example_FSM()