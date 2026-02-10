import torch
import torch.nn.functional as F    
from .utils import _cast_tensor,_fftconvolve_same,_get_device,_get_window



def cpsd(
        x,
        y,
        window_len:int,
        nfft= None,
        n_overlap:int=None, hop_len=None, # Set either n_overlap or hop_len to a value, not both
        coherence = False,
        window:str='hann',
        fs = 1.0,
        device:torch.device= None
    )->torch.Tensor:
    

        
    if (n_overlap is None and hop_len is None):
        raise ValueError("either hop len or n_overlap needs to be set")

    device = _get_device(device)

    if nfft is None:
        nfft = window_len
    
    
    with torch.no_grad():
        if hop_len is None:
            hop_len =window_len-n_overlap

        window = _get_window(window,window_len,device=device)

        if nfft>window_len:
            window = F.pad(window,(0,nfft-window_len))


        x = _cast_tensor(x,device)
        y = _cast_tensor(y,device)
        if (x.size(-1)< window_len):
            raise ValueError("Signal length needs to be larger than the window lenght")
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:   
            y = y.unsqueeze(0)
        
        n_frames = 1 + (x.size(-1) - window_len) // hop_len
        # arr_1 = torch.concat((arr_1,torch.zeros((arr_1.size(0),(nfft+hop_len*(n_frames-1))-arr_1.size(-1)),device=device,dtype=dtype)),dim=-1)
        # arr_2 = torch.concat((arr_2,torch.zeros((arr_2.size(0),(nfft+hop_len*(n_frames-1))-arr_2.size(-1)),device=device,dtype=dtype)),dim=-1)

        x = F.pad(x,(0,(nfft+hop_len*(n_frames-1))-x.size(-1)))
        y = F.pad(y,(0,(nfft+hop_len*(n_frames-1))-y.size(-1)))
        
        x_stft = torch.stft(x,win_length=window.size(-1),hop_length=hop_len,n_fft=nfft,window=window,center=False,onesided=False,return_complex=True) 
        y_stft = torch.stft(y,win_length=window.size(-1),hop_length=hop_len,n_fft=nfft,window=window,center=False,onesided=False,return_complex=True)
        # Second term conjugate following William A Gardner Statistical Spectral Analysis.
        # Same convention as matlab, conjugate of scipy.signal.csd
        out = x_stft*torch.conj(y_stft)
        out = torch.mean(out,dim=-1)
        out= out/(torch.sum(window**2)*fs) 
        if coherence:
            sxx = torch.mean(torch.abs(x_stft)**2,dim=-1)/(torch.sum(window**2)*fs) 
            syy = torch.mean(torch.abs(y_stft)**2,dim=-1)/(torch.sum(window**2)*fs) 
            out = out/(torch.sqrt(sxx*syy))
            
        # out = _cpsd_calc(x,y,nfft,window_len,window,fs,hop_len,coherence)
            
        
        
        
        if out.size(0) ==1:
            out = out.squeeze(0)
        return out
            


def cspd_fsm(
        x,
        y,
        smooth_len:int,
        fs = 1.0,
        coherence = False,
        device= None
    )->torch.Tensor:
    
    
    device = _get_device(device)
    with torch.no_grad():
        x = _cast_tensor(x,device)
        y = _cast_tensor(y,device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:   
            y = y.unsqueeze(0)
            

        
        arr_1_fft = torch.fft.fft(x)
        arr_2_fft = torch.fft.fft(y)
        fsm_est = arr_1_fft*torch.conj(arr_2_fft)

        #Manual implemented FFTConvolve using GPU, which might not be optimal for small smoothing lengths, currently chosen on a simple heuristic
        if smooth_len < 10:
            from torchaudio.functional import convolve
            fsm_est = convolve(fsm_est,torch.ones((1,smooth_len),dtype=torch.float64,device=device)/smooth_len+0j,mode='same')
        else:
            fsm_est = _fftconvolve_same(fsm_est,torch.ones((1,smooth_len),dtype=torch.float64,device=device)/smooth_len)
            
        fsm_est = fsm_est/(x.size(-1)*fs)
        if coherence:
            sxx = _fftconvolve_same(torch.abs(arr_1_fft)**2,torch.ones((1,smooth_len),dtype=torch.float64,device=device)/smooth_len)
            sxx =sxx/(x.size(-1)*fs)
            syy = _fftconvolve_same(torch.abs(arr_2_fft)**2,torch.ones((1,smooth_len),dtype=torch.float64,device=device)/smooth_len)
            syy =syy/(x.size(-1)*fs)
            fsm_est = fsm_est/(torch.sqrt(sxx*syy))
        if fsm_est.size(0) ==1:
            fsm_est = fsm_est.squeeze(0)
        return fsm_est