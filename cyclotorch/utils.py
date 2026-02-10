import torch



def _get_device(device):
    if device is None:
        try:
            device = torch.device('cuda:0')
            torch.cuda.get_device_properties(device)
        except:
            device = torch.device('cpu')
    return device

def _cast_tensor(arr,device):
    if not isinstance(arr, torch.Tensor):
        arr =  torch.tensor(arr,device=device)
    else:
        arr = arr.to(device)
    return arr

def _batch_deploy(func,out_tensor,batch_size,*args):
    
    total_samples = out_tensor.size(0)
    i=0
    while True:
        if i+batch_size>=total_samples:
            out_tensor[i:] = func(*[arg[i:] for arg in args])
            break
        out_tensor[i:i+batch_size] = func(*[arg[i:i+batch_size] for arg in args]).to(out_tensor.device)
        i += batch_size
    return out_tensor

def _get_window(type,window_length,device):
    if type is None:
        return torch.ones(window_length, device=device)
    if type.lower() == 'hann':
        return torch.hann_window(window_length, device=device)
    elif type.lower() == 'hamming':
        return torch.hamming_window(window_length, device=device)
    elif type.lower() == 'blackman':
        return torch.blackman_window(window_length, device=device)
    elif type.lower() == 'bartlett':
        return torch.bartlett_window(window_length, device=device)
    elif type.lower() == 'kaiser':
        # Default beta=12.0
        return torch.kaiser_window(window_length, periodic=True, beta=12.0, device=device)
    elif type.lower() == 'rectangular' or type.lower() == 'boxcar' or type.lower() == 'none':
        return torch.ones(window_length, device=device)
    else:
        raise ValueError(f"Unknown window type: {type}. Use one of 'hann', 'hamming', 'blackman', 'bartlett', 'kaiser' or 'rectangular'='boxcar'=None.")
    



def _complex_type(dtype):
    '''
    Returns the type for a complex number with the same precision 
    as the given dtype
    '''
    if(dtype.is_complex):
        return dtype
    else:
        return torch.view_as_complex(torch.zeros((1,2), dtype=dtype)).dtype
    
def _fftconvolve_same(arr1,arr2):
    n = arr1.size(-1)+arr2.size(-1)-1
    arr1_fft = torch.fft.fft(arr1,n)
    arr2_fft = torch.fft.fft(arr2,n)
    shift = arr2.size(-1)//2
    if arr2.size(-1) %2==0:
        shift -=1
        
    start_idx = (n - arr1.size(-1)) // 2
    return torch.fft.ifft(arr1_fft*arr2_fft,n=n)[..., start_idx : start_idx+arr1.size(-1)]


def _torch_fftconvolve_full(arr1,arr2):
    n = arr1.size(-1)+arr2.size(-1)-1
    arr1_fft = torch.fft.fft(arr1,n)
    arr2_fft = torch.fft.fft(arr2,n)
    return torch.fft.ifft(arr1_fft*arr2_fft,n=n)


