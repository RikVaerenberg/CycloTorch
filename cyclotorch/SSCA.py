import torch.nn.functional as F    
import torch 
from .utils import _cast_tensor,_get_device,_get_window

def SSCA(
        x,
        window_len:int,
        window='hann',
        fs=1.0,
        coherence = False,
        convention = 'asymmetric_negative',
        device=None
    ):
    """
    Compute the (cylic) spectral correlation using strip spectral correlation alogirthm (SSCA).

    Parameters
    ----------
    x : array-like or torch.Tensor
        Input signal. Will be flattened and cast to a torch tensor on the chosen device.
    window_len : int
        Window length used for the STFT, determines the spectral frequency resolution.
    window : str, optional
        Type of window to use (default 'hann').
    fs : float, optional
        Sampling frequency of the input signal (default 1.0).
    coherence : bool, optional
        If True, compute cyclic coherence instead of raw spectral correlation (default False).
    convention : {'asymmetric_negative', 'asymmetric_positive'}, optional
        Which CSC convention to use. Default is 'asymmetric_negative'.
    device : torch.device or str or None, optional
        Device to run computations on. If None, uses cuda device if available.

    Returns
    -------
    alpha : torch.Tensor
        Cycle frequency vector (Hz).
    f : torch.Tensor
        Spectral frequency vector (Hz), only the positive spectral frequencies are computed.
    S : torch.Tensor
        Complex spectral correlation (or coherence) array with shape (len(alpha), len(f)).

    References
    ----------
    - Borghesani, P., & Antoni, J. (2018). A faster algorithm for the calculation of the fast spectral correlation. Mechanical Systems and Signal Processing, 111, 113-118.
    - Roberts, R. S., Brown, W. A., Loomis, H. H., & H E L H, J. R. (1991). Computationally efficient algorithms for cyclic spectral analysis. IEEE Signal Processing Magazine, 8(2), 38-49.
    """



    device = _get_device(device)
    if convention not in ['asymmetric_negative','asymmetric_positive']:
        raise ValueError("convention must be one of ['asymmetric_negative' or 'asymmetric_positive']")

    with torch.no_grad():
        x = _cast_tensor(x.flatten(),device)
        window = _get_window(window,window_len,device=device)
        
        if coherence:
            x = torch.fft.irfft(torch.exp(1j * torch.angle(torch.fft.rfft(x))))
        
        n_frames = x.size(-1)
        x_pad = F.pad(x,(window_len//2,(window_len+(n_frames-1))-(x.size(-1)+window_len//2)))
        x_stft = torch.stft(x_pad,win_length=window.size(-1),n_fft=window.size(-1),hop_length=1,window=window,
                            center=False,onesided=False,return_complex=True)  #f,t

        alpha = torch.arange(n_frames,device=device)/n_frames*fs
        
        if convention == 'asymmetric_negative':
            S = torch.fft.fft(x_stft[:window.size(-1) // 2 + 1,:]*x[None,:].conj(),dim=1,norm='forward').T #alpha,f
            corr_factor =torch.exp(-1j *torch.pi *torch.arange(window_len,device=device)[:window.size(-1) // 2 + 1])[None,:] / torch.amax(window)
        elif convention == 'asymmetric_positive':
            S = torch.fft.fft(x_stft[:window.size(-1) // 2 + 1,:].conj()*x[None,:],dim=1,norm='forward').T #alpha,f
            corr_factor =torch.exp(1j *torch.pi *torch.arange(window_len,device=device)[:window.size(-1) // 2 + 1])[None,:] / torch.amax(window)
        S = S*(corr_factor) 
        
        if coherence:
            S = S / S[0,:]
        else:
            S= S/(fs)

        return alpha.cpu(),torch.abs(torch.fft.fftfreq(window.size(-1),d=1/fs,device=torch.device('cpu')))[:window.size(-1) // 2 + 1],S.cpu()
            
            
