import numpy as np
from numpy.random import default_rng
import cyclotorch
import torch
import time
from scipy.signal import sosfiltfilt,butter

rng = default_rng(seed=42)

def apply_butterbandpass(arr,lowcut,highcut,fs,order=3):
    sos_bandpassfilter = butter(N=order,Wn=[lowcut,highcut],fs=fs,output='sos',btype='bandpass')
    arr_filtered = sosfiltfilt(sos_bandpassfilter,arr)
    return arr_filtered

def main():
    # No point in running the demo if there is no GPU recognized
    # Note that GPU computation is often more efficient (at least consumer-grade) using float32 instead of float64 (and has often more than high enough precision)
    assert torch.cuda.is_available()
    Nw = 64
    alpha_max = 1/10
    
    # Generating a singal like the SineDemo
    f_envelope = 0.01
    data_size = 1_000_000
    t = np.arange(data_size)
    envelope = np.sin(t*2*np.pi*f_envelope)+1
    target = apply_butterbandpass(rng.normal(size =data_size),lowcut=0.2,highcut=0.3,fs=1.0)*envelope
    target = target*envelope
    noise = rng.normal(size =data_size)
    signal = target+noise
    signal = signal.astype(np.float32)
    target = target.astype(np.float32)

    start_time = time.time()
    alpha_arr,f,scd = cyclotorch.CSC_estimators.Fast_SC(
        signal,
        alpha_max=alpha_max,
        fs=1.0,
        window_len=Nw,
        device= torch.device('cpu')
    )
    print(f'Fast SC on the CPU took {time.time()-start_time:.3f}')
    
    
    
    # start_time = time.time()
    # _ = cyclotorch.CSC_estimators.CSC_ACP(
    #     signal,
    #     signal,
    #     alpha_arr=alpha_arr,
    #     fs=1.0,
    #     window_len=Nw,
    #     convention='asymmetric_negative',
    #     hop_len=5,
    #     device= torch.device('cpu'),
    #     per_batch=1
    # )
    # print(f'ACP on the CPU took {time.time()-start_time:.3f}')
    
    # start_time = time.time()
    # _ = cyclotorch.CSC_estimators.CSC_ACP(
    #     signal,
    #     signal,
    #     alpha_arr=alpha_arr,
    #     fs=1.0,
    #     window_len=Nw,
    #     convention='asymmetric_negative',
    #     hop_len=5,
    #     per_batch=40
    # ).cpu()
    # print(f'ACP on the GPU took {time.time()-start_time:.3f}')
    # torch.cuda.empty_cache()
    
    
    # Note that the first time a scirpt runs something on the GPU the pytorch kernels need to be initialized, which results in an increase in time
    start_time = time.time()
    alpha_arr,f,scd = cyclotorch.CSC_estimators.Fast_SC(
        signal,
        alpha_max=alpha_max,
        fs=1.0,
        window_len=Nw,
    )
    print(f'Fast SC on the GPU for the first time took {time.time()-start_time:.3f}')
    torch.cuda.empty_cache()
    start_time = time.time()
    alpha_arr,f,scd = cyclotorch.CSC_estimators.Fast_SC(
        signal,
        alpha_max=alpha_max,
        fs=1.0,
        window_len=Nw,
    )
    print(f'Fast SC on the GPU for the second time took {time.time()-start_time:.3f}')
    torch.cuda.empty_cache()
    
    start_time = time.time()
    alpha_arr,f,scd = cyclotorch.CSC_estimators.Faster_SC(
        signal,
        alpha_max=alpha_max,
        fs=1.0,
        window_len=Nw,
        device= torch.device('cpu')
    )
    print(f'Faster SC on the CPU took {time.time()-start_time:.3f}')
    
    start_time = time.time()
    alpha_arr,f,scd = cyclotorch.CSC_estimators.Faster_SC(
        signal,
        alpha_max=alpha_max,
        fs=1.0,
        window_len=Nw,
    )
    print(f'Faster SC on the GPU took {time.time()-start_time:.3f}')
    torch.cuda.empty_cache()
    
    
    start_time = time.time()
    alpha_arr,f,scd = cyclotorch.CSC_estimators.Fast_SC(
        signal,
        alpha_max=1.0,
        fs=1.0,
        window_len=Nw,
        # device= torch.device('cpu')
    )
    print(f'Fast SC (full cycle spectrum) on the GPU took {time.time()-start_time:.3f}')
    torch.cuda.empty_cache()
    
    
    start_time = time.time()
    alpha_arr,f,scd = cyclotorch.CSC_estimators.Faster_SC(
        signal,
        alpha_max=1.0,
        fs=1.0,
        window_len=Nw,
        # device= torch.device('cpu')
    )
    print(f'Faster SC (full cycle spectrum) on the GPU took {time.time()-start_time:.3f}')
    torch.cuda.empty_cache()
    
    
    start_time = time.time()
    alpha_arr,f,scd = cyclotorch.CSC_estimators.SSCA(
        signal,
        fs=1.0,
        window_len=Nw,
        # device= torch.device('cpu')

    )
    print(f'SSCA (full cycle spectrum) on the GPU took {time.time()-start_time:.3f}')
    torch.cuda.empty_cache()
    
    nr_harmonics = 10
    cycle_freq = np.array([f_envelope*i for i in range(-nr_harmonics,nr_harmonics+1)]).astype(np.float32)
    print(f'Nr of cycle frequencies considered {len(cycle_freq)}')
    start_time = time.time()
    freshfilt = cyclotorch.FRESHfilt.optimum_freshfilt(signal,target,cycle_freq=cycle_freq,filt_len=1024,fs=1.0,impose_hermitian=True,device=torch.device('cpu'))
    extracted_sig_optfresh = cyclotorch.FRESHfilt.apply_freshfilt(signal,freshfilt,device=torch.device('cpu')).data.numpy()
    print(f'Time took for Opt FRESH filtering on the CPU {time.time()-start_time:.3f}')
    
    start_time = time.time()
    freshfilt = cyclotorch.FRESHfilt.adaptive_freshfilt(signal,target,freshfilt,verbose=True,device=torch.device('cpu')) 
    extracted_sig_adaptfresh = cyclotorch.FRESHfilt.apply_freshfilt(signal,freshfilt,device=torch.device('cpu')).data.numpy()
    print(f'Time took for adapting and applying on the CPU {time.time()-start_time:.3f}')
    
    start_time = time.time()
    freshfilt = cyclotorch.FRESHfilt.optimum_freshfilt(signal,target,cycle_freq=cycle_freq,filt_len=1024,fs=1.0,impose_hermitian=True)
    extracted_sig_optfresh = cyclotorch.FRESHfilt.apply_freshfilt(signal,freshfilt).data.numpy()
    print(f'Time took for Opt FRESH filtering on the GPU {time.time()-start_time:.3f}')
    torch.cuda.empty_cache()
    
    start_time = time.time()
    freshfilt = cyclotorch.FRESHfilt.adaptive_freshfilt(signal,target,freshfilt,verbose=True) 
    extracted_sig_adaptfresh = cyclotorch.FRESHfilt.apply_freshfilt(signal,freshfilt).data.numpy()
    print(f'Time took for adapting and applying on the GPU {time.time()-start_time:.3f}')