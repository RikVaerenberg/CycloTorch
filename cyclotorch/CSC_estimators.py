import torch 
from .cpsd import cpsd,cspd_fsm
from .utils import _cast_tensor,_batch_deploy,_complex_type,_get_device
import torch.nn.functional as F    
from .Fast_SC import Fast_SC
from .Faster_SC import Faster_SC
from .SSCA import SSCA
# In order to implement these algorithms, the cyclostationary blog was a huge help: https://cyclostationary.blog/2015/12/18/csp-estimators-the-time-smoothing-method/ 



def CSC_FSM(
    x,
    y,
    alpha_arr,
    smooth_len,
    fs = 1.0,   
    convention = 'symmetric',
    coherence = False,
    device= None,
    per_batch=10 #Purely for optimization, if out of memory error occurs lower this value
    )->torch.Tensor:
    """
    Estimate the Cyclic Spectral Correlation (CSC) or Spectral Coherence using the Frequency Smoothing Method (FSM).
    
    Parameters
    ----------
    x : torch.Tensor
        Input signal (1D tensor).
    y : torch.Tensor
        Second input signal (1D tensor), must have the same length as `x`.
    alpha_arr : torch.Tensor
        Array of cyclic frequencies (1D tensor).
    smooth_len : int
        Length of the frequency smoothing window.
    fs : float, optional
        Sampling frequency of the input signals. Default is 1.0.
    convention : {'symmetric', 'asymmetric_negative', 'asymmetric_positive'}, optional
        Which CSC convention to use. Default is 'symmetric'.
    coherence : bool, optional
        If True, compute spectral coherence instead of SCD. Default is False.
    device : torch.device or None, optional
        Device to perform computation on. If None, automatically selects CUDA if available, otherwise CPU.
    per_batch : int, optional
        Number of cyclic frequencies to process per batch for optimization. Lower this value if out-of-memory errors occur. Default is 10.
    
    Returns
    -------
    torch.Tensor
       Cyclic spectral correlation or coherence tensor of shape (len(alpha_arr), len(x)).
    
    Raises
    ------
    ValueError
        If `convention` is not one of ['symmetric', 'asymmetric_negative', 'asymmetric_positive'].
        If `x` and `y` do not have the same length.
    
    Notes
    -----
    - This function implements the Frequency Smoothing Method (FSM) for cyclostationary analysis.
    - For large input sizes or cyclic frequency arrays, adjust `per_batch` to avoid memory issues.
    
    References
    -----
    - https://doi.org/10.1109/78.340776
    
    """ 
    if device is None:
        if (torch.cuda.is_available()):
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
            
    if convention not in ['symmetric','asymmetric_negative','asymmetric_positive']:
        raise ValueError("Convention must be one of ['symmetric','asymmetric_negative','asymmetric_positive']")
    if len(x)!=len(y):
        raise ValueError("x and y must have the same length")
    
    
    with torch.no_grad():
        #This could be set to GPU but will require more GPU memory
        results_device = torch.device('cpu')
        


        x = _cast_tensor(x.flatten(),results_device)
        y = _cast_tensor(y.flatten(),results_device)
        
        alpha_arr = _cast_tensor(alpha_arr,results_device)
        cdtype = _complex_type(x.dtype)

        out_tensor = torch.zeros(alpha_arr.size(0),len(x),dtype=cdtype,device=results_device)
        
        if convention == 'symmetric':
            x_alpha = x[None,:] * torch.exp(-1j*torch.pi*alpha_arr[:,None]*torch.arange(x.size(0),device=results_device)[None,:]/fs) # X(f+a/2)
            y_alpha = y[None,:] * torch.exp(1j*torch.pi*alpha_arr[:,None]*torch.arange(y.size(0),device=results_device)[None,:]/fs) # Y(f-a/2)
            batch_fsm_func = lambda batch_x,batch_y : cspd_fsm(batch_x,batch_y,smooth_len=smooth_len,fs=fs,device=device,coherence=coherence)
            out_tensor = _batch_deploy(batch_fsm_func,out_tensor,per_batch,x_alpha,y_alpha)
        elif convention == 'asymmetric_negative':
            x_alpha = x[None,:] # X(f)
            y_alpha = y[None,:] * torch.exp(2j*torch.pi*alpha_arr[:,None]*torch.arange(y.size(0),device=results_device)[None,:]/fs) # Y(f-a)
            batch_fsm_func = lambda batch_y : cspd_fsm(x_alpha,batch_y,smooth_len=smooth_len,fs=fs,device=device,coherence=coherence)
            out_tensor = _batch_deploy(batch_fsm_func,out_tensor,per_batch,y_alpha)
        if convention == 'asymmetric_positive':
            x_alpha = x[None,:] * torch.exp(-2j*torch.pi*alpha_arr[:,None]*torch.arange(x.size(0),device=results_device)[None,:]/fs) # X(f+a)
            y_alpha = y[None,:]  # Y(f)
            batch_fsm_func = lambda batch_x : cspd_fsm(batch_x,y_alpha,smooth_len=smooth_len,fs=fs,device=device,coherence=coherence)
            out_tensor = _batch_deploy(batch_fsm_func,out_tensor,per_batch,x_alpha)
        return out_tensor



def CSC_ACP(
        x, 
        y,
        alpha_arr,
        window_len:int,
        nfft= None,
        n_overlap:int=None, hop_len=None, # Set either n_overlap or hop_len to a value, not both
        window='hann',
        fs = 1.0,
        convention = 'symmetric',
        coherence = False,
        device= None,
        results_device = torch.device('cpu'), #Can be set to GPU, but requires more GPU memory
        per_batch = 20 #Purely for optimization, if out of memory error occurs lower this value
        
    )->torch.Tensor:
    """
    Estimate the Cyclic Spectral Correlation (CSC) or Coherence using the Average Cyclic Periodogram (ACP) estimator.
    
    Parameters
    ----------
    x : torch.Tensor or array-like
        Input signal to analyze. Will be flattened and cast to a torch tensor.
    y : torch.Tensor or array-like
        Second input signal (for cross-spectral analysis). Will be flattened and cast to a torch tensor. 
    alpha_arr : torch.Tensor or array-like
        Array of cycle frequencies (Hz) to compute.
    window_len : int
        Length of the window function to use for STFT.
    nfft : int or None, optional
        Number of FFT points. If None, defaults to window_len.
    n_overlap : int or None, optional
        Number of overlapping samples between windows. If None, hop_len must be set.
    hop_len : int or None, optional
        Hop size between windows. If None, n_overlap must be set.
    window : str, optional
        Type of window to use. Default is 'hann'.
    fs : float, optional
        Sampling frequency of the input signal. Default is 1.0.
    convention : {'symmetric', 'asymmetric_negative', 'asymmetric_positive'}, optional
        Which CSC convention to use. Default is 'symmetric'.
    coherence : bool, optional
        If True, computes the coherence instead of the correlation. Default is False.
    device : torch.device or str or None, optional
        Device to perform computation on. If None, uses cuda device if available.
    results_device : torch.device, optional
        Device to store results. Default is CPU. Can be set to GPU for faster computation (requires more memory).
    per_batch : int, optional
        Batch size for computation. Lower if out-of-memory errors occur. Default is 20.

    Returns
    -------
    out_tensor : torch.Tensor
        Spectral correlation (or coherence), shape (len(alpha_arr), nfft) this array is still on the given device.

    Raises
    ------
    ValueError
        If `convention` is not one of ['symmetric', 'asymmetric_negative', 'asymmetric_positive'].

    Notes
    -----
    - Batch processing is used for memory efficiency; adjust per_batch for large signals or limited GPU memory.
    
    References
    -----
    - https://doi.org/10.1109/78.340776
    
    """
    

    device = _get_device(device)
            
    if convention not in ['symmetric','asymmetric_negative','asymmetric_positive']:
        raise ValueError("Convention must be one of ['symmetric','asymmetric_negative','asymmetric_positive']")
    
    with torch.no_grad():
        
        if nfft is None:
            nfft = window_len
        
        x = _cast_tensor(x.flatten(),results_device)
        y = _cast_tensor(y.flatten(),results_device)
        alpha_arr = _cast_tensor(alpha_arr,results_device)
        cdtype = _complex_type(x.dtype)

        out_tensor = torch.zeros(alpha_arr.size(0),nfft,dtype=cdtype,device=results_device)
        
        if convention == 'symmetric':
            batch_cspd_func = lambda alpha_arr_b : cpsd(
                x[None,:] * torch.exp(-1j*torch.pi*alpha_arr_b[:,None]*torch.arange(x.size(0),device=results_device)[None,:]/fs), # X(f+a/2),
                y[None,:] * torch.exp(1j*torch.pi*alpha_arr_b[:,None]*torch.arange(y.size(0),device=results_device)[None,:]/fs), # Y(f-a/2),
                window_len,nfft=nfft,n_overlap=n_overlap,hop_len=hop_len,window=window,fs=fs,device=device,coherence=coherence)
        elif convention == 'asymmetric_negative':

            batch_cspd_func = lambda alpha_arr_b : cpsd(x[None,:],# X(f)
                                                        y[None,:] * torch.exp(2j*torch.pi*alpha_arr_b[:,None]*torch.arange(y.size(0),device=results_device)[None,:]/fs),# Y(f-a)
                                                        window_len,nfft=nfft,n_overlap=n_overlap,hop_len=hop_len,window=window,fs=fs,device=device,coherence=coherence)
        elif convention == 'asymmetric_positive':

            batch_cspd_func = lambda alpha_arr_b : cpsd(x[None,:] * torch.exp(-2j*torch.pi*alpha_arr_b[:,None]*torch.arange(x.size(0),device=results_device)[None,:]/fs), # X(f+a)
                                                        y[None,:], # Y(f)
                                                        window_len,nfft=nfft,n_overlap=n_overlap,hop_len=hop_len,window=window,fs=fs,device=device,coherence=coherence)
        out_tensor = _batch_deploy(batch_cspd_func,out_tensor,per_batch,alpha_arr)
        return out_tensor

        
