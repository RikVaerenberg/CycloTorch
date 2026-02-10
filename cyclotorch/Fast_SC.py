import torch.nn.functional as F    
import torch 
from .utils import _cast_tensor,_get_device,_get_window
from math import ceil
# Orignal algorithm by J.Antoni code adapted and written by Rik Vaerenberg



def Fast_SC(
        x,
        alpha_max,
        window_len:int,
        window='hann',
        fs=1.0,
        R = None,
        coherence = False,
        convention = 'asymmetric_positive', # Following the paper of Jerome Antoni 
        device=None
    ):
    """
    Compute the (cylic) spectral correlation using the fast spectral correlation method of J.Antoni.
    
    Parameters
    ----------
    
    x : torch.Tensor or array-like
        Input signal to analyze. Will be flattened and cast to a torch tensor.
    alpha_max : float
        Maximum cycle frequency to compute (Hz).
    window_len : int
        Length of the window function to use for STFT.
    window : str, optional
        Type of window to use.
    fs : float, optional
        Sampling frequency of the input signal. Default is 1.0.
    R : int or None, optional
        Hop size between windows. If None, it is automatically determined to obtain the required alpha_max and have at least 75% overlapping windows.
    coherence : bool, optional
        If True, computes the coherence instead of the correlation. Default is False.
    convention : {'symmetric', 'asymmetric_negative', 'asymmetric_positive'}, optional
        Which CSC convention to use. Default is 'asymmetric_positive'.
    device : torch.device or str or None, optional
        Device to perform computation on. If None, uses cuda device if available.

    Returns
    -------
    
    alpha : torch.Tensor
        Array of cycle frequencies (Hz) up to `alpha_max`.
    f : torch.Tensor
        Array of spectral frequencies (Hz), for the symmetric convention the spectral resolution is doubled
    S : torch.Tensor
        Spectral correlation (or coherence), shape (len(alpha), len(f)).
    
    Raises
    ------
    ValueError
        If `convention` is not one of ['symmetric', 'asymmetric_negative', 'asymmetric_positive'].
    ValueError
        If the window is not recognized
    
    Notes
    -----
    - For cylic spectral coherence estimation, additional prewhitening is performed for robustness with respect to the matlab code of Jerome.
    
    References
    -----
    - https://doi.org/10.1016/j.ymssp.2017.01.011
    
    """


    device = _get_device(device)
    if convention not in ['symmetric','asymmetric_negative','asymmetric_positive']:
        raise ValueError("convention must be one of ['symmetric'','asymmetric_negative' or 'asymmetric_positive']")
    if R is None:
        R = int(fs/2/alpha_max)
        R = max(1,min(R,int(window_len/4)))
    with torch.no_grad():
        x = _cast_tensor(x.flatten(),device)

        window = _get_window(window,window_len,device=device)

        if convention == 'symmetric':
            # Best to use an even window lenght to retain full window symmetry
            window = F.pad(window,(window_len//2,window_len//2+window_len%2))
            nfft = window_len*2
        else:
            nfft = window_len
        n_frames = 1 + (x.size(-1) - window_len) // R
        if coherence:
            # Additional prewhitening needed to have robust coherence estimate
            x = torch.fft.irfft(torch.exp(1j * torch.angle(torch.fft.rfft(x))))
        x = F.pad(x,(0,(nfft+R*(n_frames-1))-x.size(-1)))
        x_stft = torch.stft(x,win_length=window.size(-1),hop_length=R,n_fft=nfft,window=window,center=False,onesided=False,return_complex=True)  #f,t
        
        max_p = int(ceil(window_len/(2*R)))
        max_p = max(1,max_p)
        window_sq = window**2
        
        # Assuming symmetric window with max at the middle
        midpoint = (nfft-1)/2
        if coherence:
            x_stft = x_stft/torch.sqrt(torch.mean(torch.abs(x_stft)**2,dim=-1))[:,None]
            
        #Abs for solving the issue of the 0.5*fs being negative in case of even amount of samples
        alpha = torch.abs(torch.fft.fftfreq(x_stft.size(1),d=R/fs,device=device)[ :x_stft.size(1) // 2 + 1]) 
        alpha_max_idx = torch.argwhere(alpha>alpha_max)[0] if len(torch.argwhere(alpha>alpha_max))>0 else len(alpha)
        alpha = alpha[:alpha_max_idx]
        S = _cps_shifted_neg(x_stft,0)[:alpha_max_idx,:]
        S = S*torch.exp(-2j*torch.pi*midpoint*(alpha[:,None])/fs) 
        R_correct = _R_w(window_sq,alpha=alpha,fs=fs)
        
        # Note that in the orignal paper, only the positive p's were considered. But in the matlab implementation both negative and positive ones are considered
        for p in range(-max_p, max_p+1):
            if p ==0:
                continue
            shift_freq = p/window_len*fs
            if convention == 'asymmetric_positive':
                cps_shift = _cps_shifted_pos(x_stft,p)[:alpha_max_idx,:]
            elif convention == 'asymmetric_negative':
                cps_shift = _cps_shifted_neg(x_stft,p)[:alpha_max_idx,:]
            else:
                cps_shift = _cps_shifted_symm(x_stft,p)[:alpha_max_idx,:]
            #Note we do not update the zero cycle freq as this results in numerical instabilities when computing the coherence
            R_correct[1:] += _R_w(window_sq,alpha=alpha-shift_freq,fs=fs)[1:]
            S[1:,:] += (cps_shift*torch.exp(-2j*torch.pi*midpoint*(alpha[:,None]-shift_freq)/fs))[1:,:] # Phase correction
        S = S/R_correct[:,None]/fs
        if coherence:
            S = S*R_correct[0]*fs
        return alpha.cpu(),torch.abs(torch.fft.fftfreq(nfft,d=1/fs,device=torch.device('cpu')))[:nfft // 2 + 1],S.cpu()
            
            

def _R_w(window_sq,alpha,fs):
    midpoint = (window_sq.size(0)-1)/2
    return torch.sum(window_sq[:,None]*\
        torch.exp(-2j*torch.pi*(torch.arange(window_sq.size(0),device=window_sq.device)[:,None]-midpoint)*alpha[None,:]/fs),dim=0)

def _cps_shifted_neg(stft_x,shift_amount):
    cps = stft_x[:stft_x.size(0) // 2 + 1]*torch.roll(stft_x,shift_amount,0)[:stft_x.size(0) // 2 + 1].conj()
    cps = torch.fft.fft(cps,dim=1,norm='forward')[:, :cps.size(1) // 2 + 1].T
    return cps

def _cps_shifted_pos(stft_x,shift_amount): 
    cps = torch.roll(stft_x,-shift_amount,0)[:stft_x.size(0) // 2 + 1]*stft_x[:stft_x.size(0) // 2 + 1].conj() #f,t
    cps = torch.fft.fft(cps, dim=1, norm='forward')[:, :cps.size(1) // 2 + 1].T
    # cps = torch.fft.rfft(cps.real, dim=1, norm='forward').T+ 1j*torch.fft.rfft(cps.imag, dim=1, norm='forward').T
    return cps

def _cps_shifted_symm(stft_x,shift_amount):
    cps =  torch.roll(stft_x,-shift_amount,0)[:stft_x.size(0) // 2 + 1]*torch.roll(stft_x,shift_amount,0)[:stft_x.size(0) // 2 + 1].conj()
    cps = torch.fft.fft(cps,dim=1,norm='forward')[:, :cps.size(1) // 2 + 1].T
    return cps
