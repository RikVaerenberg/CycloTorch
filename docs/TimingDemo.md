# Timing Demo

This demo illustrates the benefit of running cyclostationary analysis on the GPU. Note that a CUDA device is required.[Code for full demo](https://gitlab.kuleuven.be/lmsd-cm/dsp-python/cyclostationary-analysis/torchcyclostationarydev/-/blob/main/demos/TimingDemo.py?ref_type=heads)



---

## 1. Generating a cyclostationary signal

Bandpass filtering:
```python
def apply_butterbandpass(arr, lowcut, highcut, fs, order=3):
	sos_bandpassfilter = butter(N=order, Wn=[lowcut, highcut], fs=fs, output='sos', btype='bandpass')
	arr_filtered = sosfiltfilt(sos_bandpassfilter, arr)
	return arr_filtered
```

Generate a sine-modulated envelope and use it to modulate Gaussian noise:
```python
rng = default_rng(seed=42)
data_size = 1_000_000
f_envelope = 0.01
t = np.arange(data_size)
envelope = np.sin(t * 2 * np.pi * f_envelope) + 1
target = apply_butterbandpass(rng.normal(size=data_size), lowcut=0.2, highcut=0.3, fs=1.0) * envelope
target = target * envelope
noise = rng.normal(size=data_size)
signal = target + noise
signal = signal.astype(np.float32)
target = target.astype(np.float32)
```

They are casted to float32 for efficiency on the GPU


## 2. Time comparison across Spectral Correlation Algorithms

Run Fast_SC and Faster_SC on CPU and GPU, timing their execution:
```python
import torchcyclostationary as tc
import torch
import time
Nw = 128
alpha_max = 1/10

# Fast_SC on CPU
start_time = time.time()
alpha_arr, f, scd = tc.CSC_estimators.Fast_SC(
	signal,
	alpha_max=alpha_max,
	fs=1.0,
	window_len=Nw,
	device=torch.device('cpu')
)
print(f'Fast SC on the CPU took {time.time()-start_time:.3f}')

# Fast_SC on GPU (first and second run)
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
# Faster_SC on CPU
start_time = time.time()
alpha_arr, f, scd = tc.CSC_estimators.Faster_SC(
	signal,
	alpha_max=alpha_max,
	fs=1.0,
	window_len=Nw,
	device=torch.device('cpu')
)
print(f'Faster SC on the CPU took {time.time()-start_time:.3f}')

# Faster_SC on GPU
start_time = time.time()
alpha_arr, f, scd = tc.CSC_estimators.Faster_SC(
	signal,
	alpha_max=alpha_max,
	fs=1.0,
	window_len=Nw,
)
print(f'Faster SC on the GPU took {time.time()-start_time:.3f}')
torch.cuda.empty_cache()
```

## 2.1 Full-cycle timings
Run Fast_SC, Faster_SC (for the full cycle frequency range) and the SSCA, timing their execution:

```python
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
```

## 3. Timing FRESH Filtering Algorithms

Run optimum and adaptive FRESH filtering on CPU and GPU, timing their execution:
```python
nr_harmonics = 10
cycle_freq = np.array([f_envelope*i for i in range(-nr_harmonics,nr_harmonics+1)]).astype(np.float32)
print(f'Nr of cycle frequencies considered {len(cycle_freq)}')

# Optimum FRESH filtering on CPU
start_time = time.time()
freshfilt = tc.FRESHfilt.optimum_freshfilt(signal, target, cycle_freq=cycle_freq, filt_len=1024, fs=1.0, impose_hermitian=True, device=torch.device('cpu'))
extracted_sig_optfresh = tc.FRESHfilt.apply_freshfilt(signal, freshfilt, device=torch.device('cpu')).data.numpy()
print(f'Time took for Opt FRESH filtering on the CPU {time.time()-start_time:.3f}')

# Adaptive FRESH filtering on CPU
start_time = time.time()
freshfilt = tc.FRESHfilt.adaptive_freshfilt(signal, target, freshfilt, verbose=True, device=torch.device('cpu'))
extracted_sig_adaptfresh = tc.FRESHfilt.apply_freshfilt(signal, freshfilt, device=torch.device('cpu')).data.numpy()
print(f'Time took for adapting and applying on the CPU {time.time()-start_time:.3f}')

# Optimum FRESH filtering on GPU
start_time = time.time()
freshfilt = tc.FRESHfilt.optimum_freshfilt(signal, target, cycle_freq=cycle_freq, filt_len=1024, fs=1.0, impose_hermitian=True)
extracted_sig_optfresh = tc.FRESHfilt.apply_freshfilt(signal, freshfilt).data.numpy()
print(f'Time took for Opt FRESH filtering on the GPU {time.time()-start_time:.3f}')

# Adaptive FRESH filtering on GPU
start_time = time.time()
freshfilt = tc.FRESHfilt.adaptive_freshfilt(signal, target, freshfilt, verbose=True)
extracted_sig_adaptfresh = tc.FRESHfilt.apply_freshfilt(signal, freshfilt).data.numpy()
print(f'Time took for adapting and applying on the GPU {time.time()-start_time:.3f}')
```

---
