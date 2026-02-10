# Sine Demo

This demo illustrates the workflow for generating a cyclostationary signal, adding noise at a controlled SNR, analyzing its spectral properties, and applying FRESH filtering for signal extraction. [Code for full demo](https://gitlab.kuleuven.be/lmsd-cm/dsp-python/cyclostationary-analysis/torchcyclostationarydev/-/blob/main/demos/SineDemo.py?ref_type=heads)

---

## 1. Helper Functions

SNR scaling, bandpass filtering, and SNR computation:
```python
def scale_snr(signal, noise, target_snr_db):
	signal_power = np.mean(signal**2)
	noise_power = np.mean(noise**2)
	noise_scaling_factor = np.sqrt(signal_power / (noise_power*10**(target_snr_db/10)))
	scaled_noise = noise * noise_scaling_factor
	return signal + scaled_noise, noise_scaling_factor

def apply_butterbandpass(arr, lowcut, highcut, fs, order=3):
	sos_bandpassfilter = butter(N=order, Wn=[lowcut, highcut], fs=fs, output='sos', btype='bandpass')
	arr_filtered = sosfiltfilt(sos_bandpassfilter, arr)
	return arr_filtered

def compute_snr(measured_signal, target_signal):
	noise = measured_signal - target_signal
	signal_power = np.mean(target_signal ** 2)
	noise_power = np.mean(noise ** 2)
	snr_db = 10 * np.log10(signal_power / noise_power)
	return snr_db
```
---

## 2. Signal Generation and Envelope Modulation

Generate a sine-modulated envelope and use it to modulate Gaussian noise:
```python
rng = default_rng(seed=42)
data_size = 100_000
f_envelope = 0.01
t = np.arange(data_size)
envelope = np.sin(t * 2 * np.pi * f_envelope) + 1
```
---

## 3. Target Signal and Noise Construction

Create a bandpass-filtered target and add noise at a specified SNR:
```python
target = apply_butterbandpass(rng.normal(size=data_size), lowcut=0.2, highcut=0.3, fs=1.0) * envelope
noise = rng.normal(size=data_size)
signal, noise_scale = scale_snr(target, noise, snr=5)
noise = noise * noise_scale
print(f'SNR before filtering {compute_snr(signal, target)}')
```
---

## 4. Cyclostationary Spectral Analysis

Compute the cyclic spectral correlation (CSC) and coherence using the Faster_SC estimator:
```python
import torchcyclostationary as tc
alpha_arr, f, scd = tc.CSC_estimators.Faster_SC(
	signal,
	alpha_max=0.05,
	fs=1.0,
	window_len=256,
)
alpha_arr, f, scd = alpha_arr.numpy(), f.numpy(), scd.numpy()

# Coherence
alpha_arr, f, coh = tc.CSC_estimators.Faster_SC(
	signal,
	alpha_max=0.05,
	fs=1.0,
	window_len=256,
	coherence=True
)
alpha_arr, f, coh = alpha_arr.numpy(), f.numpy(), coh.numpy()
```
---

## 5. FRESH Filter optimization and Application

Compute and apply the optimum and adaptive FRESH filters to extract the target signal:
```python
cycle_freq = np.array([-f_envelope, 0, f_envelope])
freshfilt = tc.FRESHfilt.optimum_freshfilt(signal, target, cycle_freq=cycle_freq, filt_len=1024, fs=1.0, impose_hermitian=True)
extracted_sig_optfresh = tc.FRESHfilt.apply_freshfilt(signal, freshfilt).data.numpy()
print(f'SNR after FRESH filtering {compute_snr(extracted_sig_optfresh, target)}')
freshfilt = tc.FRESHfilt.adaptive_freshfilt(signal, target, freshfilt, verbose=True)
extracted_sig_adaptfresh = tc.FRESHfilt.apply_freshfilt(signal, freshfilt).data.numpy()
print(f'obtained snr after filtering using adaptive freshfilt {compute_snr(extracted_sig_adaptfresh, target)}')
```
---

