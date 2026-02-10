import torch
from .utils import _cast_tensor,_complex_type,_fftconvolve_same,_get_device,_torch_fftconvolve_full
from .CSC_estimators import CSC_ACP
# from .SCD_estimators import _cross_alpha_density
from torchaudio.functional import convolve

class FRESHFilter:
    def __init__(self,cycle_freq,
                 filterbank,fs):
        self.fs = fs
        self.cycle_freq = _cast_tensor(cycle_freq,device=torch.device('cpu'))
        self.filterbank = _cast_tensor(filterbank,device=torch.device('cpu'))


def _check_symmetry_cyclefreq(cycle_freq):
    for i in range(len(cycle_freq)):
        if not  cycle_freq[i]== -cycle_freq[-1-i]:
            raise ValueError("If the hermitian is imposed the cycle frequency array should be sorted, and contain the positve and negative cycle frequencies")


def optimum_freshfilt(signal,
                      target,
                      cycle_freq,
                      filt_len,
                      fs,
                      impose_hermitian = False, # Note that in this case the cycle freq should be symmetric around zero (eg cycle_freq[i]= -cycle_freq[-1-i] for all i)
                      device=None,
                      per_batch =10 #Higher = quicker but more memory intensive
                      ):  

    """
    Compute the optimum FRESH (Frequency Shift) filter for a given signal and target signal.
    
    Parameters
    ----------
    signal : array-like or torch.Tensor
        Real input signal to be filtered. Should be 1D.
    target : array-like or torch.Tensor
        Real target signal for filter optimization. Should be 1D and of the same length as `signal`.
    cycle_freq : array-like or torch.Tensor
        Array of cycle frequencies (in Hz) to use for cyclostationary analysis.
    filt_len : int
        Length of the filter.
    fs : float
        Sampling frequency of the signals (in Hz).
    impose_hermitian : bool, optional
        If True, enforces Hermitian symmetry on the cycle frequencies and the resulting filter.
        Note: `cycle_freq` should be symmetric around zero if this is set to True.
    device : torch.device or str, optional
        Device to perform computations on (e.g., 'cpu', 'cuda'). If None, automatically selected.
    per_batch : int, optional
        Number of batches for computation. Higher values increase speed but use more memory.
    
    Returns
    -------
    FRESHFilter
        An instance of FRESHFilter containing the computed filter coefficients and associated parameters.
    
    Raises
    ------
    ValueError
        If impose Hermetian is true and the cycle frequencies are not symmetric around zero.
    
    Notes
    -----
    - The function uses cyclostationary cross-correlation and auto-correlation matrices for filter estimation.

    """
    
    
    device = _get_device(device)
    
    if impose_hermitian:
        _check_symmetry_cyclefreq(cycle_freq)
    
    with torch.no_grad():
        signal = _cast_tensor(signal,device=device)
        target = _cast_tensor(target,device=device)
        cycle_freq = _cast_tensor(cycle_freq,device=device)
        
        S_matrix = torch.zeros(len(cycle_freq),len(cycle_freq),filt_len,dtype=_complex_type(signal.dtype))
        signal_cycle = signal[None,:] *torch.exp(2j*torch.pi*cycle_freq[:,None]*torch.arange(len(signal),device=device)/fs)
        for i in range(S_matrix.size(0)):
            S_matrix[i,:,:] = CSC_ACP(signal_cycle[i,:],
                                     signal,cycle_freq,
                                     window='kaiser',
                                     window_len=filt_len,
                                     n_overlap=int(0.75*filt_len),
                                     fs=fs,convention='asymmetric_negative',
                                     per_batch=per_batch,
                                     device=device,
                                     results_device=device,
                                     coherence=False)
        # S_matrix is now a1,a2,f should be f,a2,a1
        S_matrix = torch.permute(S_matrix,(2,1,0))
        
        
        if impose_hermitian:
            S_matrix = (S_matrix + torch.transpose(S_matrix,
                                                   1,2).conj())/2
        B_matrix = CSC_ACP(signal,target,cycle_freq,
                                    window_len=filt_len,
                                    window='kaiser',
                                    n_overlap=int(0.75*filt_len), 
                                    fs=fs,
                                    convention='asymmetric_negative',
                                    per_batch=per_batch,
                                    device=device,
                                    results_device=device,
                                    coherence=False).T
        
        # Done on CPU as torch linalg GPU requires lower precision to be quick (and synchronizes with the CPU anyway)
        filterval =torch.linalg.solve(S_matrix.cpu(),B_matrix.cpu()).T #a,f
        return FRESHFilter(cycle_freq,filterval,fs)


def adaptive_freshfilt(signal,
                       target,
                       freshfilt: FRESHFilter,
                       max_iter = 300,
                       verbose = False,
                       device=None
                       ):
    """
    Adaptively optimizes the FRESH filter coefficients to minimize the mean squared error between the filtered signal and a target signal.
    
    Parameters
    ----------
    signal : torch.Tensor or array-like
        Input signal to be filtered.
    target : torch.Tensor or array-like
        Target signal to match after filtering.
    freshfilt : FRESHFilter
        Instance of FRESHFilter containing initial filterbank and cycle frequencies.
    max_iter : int, optional
        Maximum number of optimization iterations (default: 300).
    verbose : bool, optional
        If True, prints optimization progress and timing information (default: False).
    device : torch.device or str, optional
        Device on which to perform computations. If None, uses default device.
    
    Returns
    -------
    FRESHFilter
        A new FRESHFilter instance with optimized filterbank coefficients.
    
    Notes
    -----
    - The function uses the LBFGS optimizer to adapt the real and imaginary parts of the filterbank coefficients.

    """
    
    
    
    
    
    
    device = _get_device(device)
    
    signal = _cast_tensor(signal,device=device)
    target = _cast_tensor(target,device=device)
    cycle_freq = _cast_tensor(freshfilt.cycle_freq,device=device)
    filterbank_real = _cast_tensor(torch.real(freshfilt.filterbank).detach().clone(),device=device).requires_grad_(True)
    filterbank_imag = _cast_tensor(torch.imag(freshfilt.filterbank).detach().clone(),device=device).requires_grad_(True)
    optimizer = torch.optim.LBFGS([filterbank_real,filterbank_imag],max_iter=max_iter,line_search_fn='strong_wolfe',history_size=30)
    filt_len = freshfilt.filterbank.size(-1)
    iteration = 0
    def closure():
        nonlocal iteration
        iteration +=1
        optimizer.zero_grad()
        
        arr_cycle = signal[None,:]*torch.exp(2j*torch.pi*cycle_freq[:,None]*torch.arange(len(signal),device=device)[None,:]/freshfilt.fs)
        filt_arr = torch.fft.ifft(filterbank_real+1j*filterbank_imag,dim=-1,norm='backward')
        
        shift = filt_arr.size(-1)//2
        # Center around tau=0 
        filt_arr = torch.fft.fftshift(filt_arr,dim=-1)
        if filt_len < 10:
            
            out = torch.sum(convolve(arr_cycle,filt_arr,mode='full')[:,shift:len(signal)+shift],dim=0)
        else:
            out = torch.sum(_torch_fftconvolve_full(arr_cycle,filt_arr)[:,shift:len(signal)+shift],dim=0)
            
        loss = torch.mean(torch.abs(out-target)**2)
        loss.backward()
        return loss
    from time import time
    start_time= time()
    optimizer.step(closure)
    if verbose:
        print(f'optimizer took {time()-start_time:.3f}s and called {iteration} times (of a set maximum iterations {max_iter})')
    return FRESHFilter(cycle_freq,filterbank_real+1.0j*filterbank_imag,freshfilt.fs)




def apply_freshfilt(signal,freshfilt:FRESHFilter,device=None):
    """
    Apply a FRESH filter to a real 1D input signal.
    
    Parameters
    ----------
    signal : torch.Tensor
        1D input signal to be filtered.
    freshfilt : FRESHFilter
        FRESHFilter object containing filterbank, cycle frequencies, and sampling frequency.
    device : torch.device or str, optional
        Device on which to perform computation (e.g., 'cpu', 'cuda'). If None, uses default device.

    Returns
    -------
    torch.Tensor
        The real filtered signal as a 1D tensor on CPU.
    
    Notes
    -----
    - There is shifting applied to the filterbank in order to approach zero-phase filtering.
    - For short filters (length < 10), uses `torchaudio.functional.convolve`; otherwise, uses FFT-based convolution.
    - The output is always moved to CPU and contains only the real part of the filtered signal.

    """

    
    
    filt_len = freshfilt.filterbank.size(-1)
    device = _get_device(device)
    with torch.no_grad():
        signal = _cast_tensor(signal,device)
        cycle_freq =_cast_tensor(freshfilt.cycle_freq,device)
        # out_arr =torch.zeros_like(arr,dtype=_complex_type(arr))
        sig_cycle = signal[None,:]*torch.exp(2j*torch.pi*cycle_freq[:,None]*torch.arange(len(signal),device=device)[None,:]/freshfilt.fs)
        filt_arr = torch.fft.ifft(_cast_tensor(freshfilt.filterbank,device),dim=-1,norm='backward')
        
        shift = filt_arr.size(-1)//2
        # Center around tau=0 
        filt_arr = torch.fft.fftshift(filt_arr,dim=-1)
        if filt_len < 10:
            from torchaudio.functional import convolve
            out = torch.sum(convolve(sig_cycle,filt_arr,mode='full')[:,shift:len(signal)+shift],dim=0)
        else:
            out = torch.sum(_torch_fftconvolve_full(sig_cycle,filt_arr)[:,shift:len(signal)+shift],dim=0)
        return torch.real(out).cpu()


