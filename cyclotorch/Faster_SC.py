import torch.nn.functional as F    
import torch 
from .utils import _cast_tensor,_get_device,_get_window
from math import ceil
# Orignal algorithm by P.Borghesani code adapted and written by Rik Vaerenberg

def Faster_SC(
        x,
        alpha_max,
        window_len:int,
        window='hann',
        fs=1.0,
        R = None,
        coherence = False,
        convention = 'asymmetric_negative',
        device=None
    ):
    """
    Compute the (cylic) spectral correlation using the faster spectral correlation method by P.Borghesani.
    
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
    convention : {'asymmetric_negative', 'asymmetric_positive'}, optional
        Which CSC convention to use. Default is 'asymmetric_negative'.
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
        If `convention` is not one of ['asymmetric_negative', 'asymmetric_positive'].
    ValueError
        If the window is not recognized
    
    Notes
    -----
    - The coherence computation is done based on the prewhitened array instead of the proposed method by P.Borghesani for numerical stability.
    
    References
    -----
    - https://doi.org/10.1016/j.ymssp.2018.03.059
    
    """


    device = _get_device(device)
    if convention not in ['asymmetric_negative','asymmetric_positive']:
        raise ValueError("convention must be one of ['asymmetric_negative' or 'asymmetric_positive']")
    if R is None:
        R = int(fs/2/alpha_max)
        R = max(1,min(R,int(window_len/4)))
    with torch.no_grad():
        x = _cast_tensor(x.flatten(),device)
        window = _get_window(window,window_len,device=device)
        
        max_p = int(ceil(window_len/(2*R)))
        max_p = max(1,max_p)
        p_tens = torch.arange(-max_p,max_p+1,device=device)
        if convention =='asymmetric_negative':
            neg_corr = torch.sum(
                torch.exp(2j*torch.pi*(torch.arange(window_len,device=device)[:,None]-window_len/2)/window_len*p_tens[None,:]),
                dim=-1)
            window_1 = window
            window_2 = window*neg_corr
        else:
            pos_corr = torch.sum(
                torch.exp(-2j*torch.pi*(torch.arange(window_len,device=device)[:,None]-window_len/2)/window_len*p_tens[None,:]),
                dim=-1)
            window_1 = window *pos_corr
            window_2 = window
        if coherence:
            # Additional prewhitening needed to have robust coherence estimate
            x = torch.fft.irfft(torch.exp(1j * torch.angle(torch.fft.rfft(x))))
        
        n_frames = 1 + (x.size(-1) - window_len) // R
        x = F.pad(x,(0,(window_len+R*(n_frames-1))-x.size(-1)))
        x_stft_1 = torch.stft(x,win_length=window.size(-1),n_fft=window.size(-1),hop_length=R,window=window_1,center=False,onesided=False,return_complex=True)  #f,t
        x_stft_2 = torch.stft(x,win_length=window.size(-1),n_fft=window.size(-1),hop_length=R,window=window_2,center=False,onesided=False,return_complex=True)  #f,t


        alpha = torch.abs(torch.fft.fftfreq(x_stft_1.size(1),d=R/fs,device=device)) [ :x_stft_1.size(1) // 2 + 1]

        S = torch.fft.fft(x_stft_1[:x_stft_1.size(0) // 2 + 1]*x_stft_2[:x_stft_1.size(0) // 2 + 1].conj(),dim=1,norm='forward')[:, :x_stft_1.size(1) // 2 + 1].T
        
        corr_factor = torch.fft.fft(window_1.conj()*window_2,n =x_stft_1.size(1)*R).conj() #Additional conj from error in derivation of Jerome
        #Abs for solving the issue of the 0.5*fs being negative in case of even amount of samples
        alpha_max_idx = torch.argwhere(alpha>alpha_max)[0] if len(torch.argwhere(alpha>alpha_max))>0 else len(alpha)
        alpha = alpha[:alpha_max_idx]
        S = S[:alpha_max_idx]
        corr_factor = corr_factor[:alpha_max_idx]
        S = S/(corr_factor[:,None]) 
        
        if coherence:
            # Originally proposed in the paper, but very unstable. Better to follow the prewhitening technique as proposed by J.Antoni
            # if convention =='asymmetric_negative':
            #     idx = torch.arange(S.size(1))[None,:]-torch.arange(S.size(0))[:,None]/(x_stft_1.size(1)*R)*window_len
            # else:
            #     idx = torch.arange(S.size(1))[None,:]+torch.arange(S.size(0))[:,None]/(x_stft_1.size(1)*R)*window_len
            # idx = idx.to(torch.int)
            # idx = idx%S.size(1)
            # S_fminusalpha = S[0,:].flatten()[idx]
            # S = S/torch.sqrt(S[0,:][None,:]*S_fminusalpha)
            # S = torch.clamp(S.abs(), 0, 1) * torch.exp(1j * S.angle()) 
            S = S / S[0,:]
        else:
            S= S/(fs)

        return alpha.cpu(),torch.abs(torch.fft.fftfreq(window.size(-1),d=1/fs,device=torch.device('cpu')))[:window.size(-1) // 2 + 1],S.cpu()
            
            
