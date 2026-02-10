# One Degree of Freedom (1DoF) Demo

This demo illustrates the workflow for generating a cyclostationary excitation, simulating a single degree-of-freedom (SDOF) system response, adding noise at a controlled SNR, and applying FRESH filtering for signal extraction. [Code for full demo](https://gitlab.kuleuven.be/lmsd-cm/dsp-python/cyclostationary-analysis/torchcyclostationarydev/-/blob/main/demos/OneDoFDemo.py?ref_type=heads)


---

## 1. Helper Functions

SNR scaling and SNR computation:
```python
def scale_snr(signal, noise, target_snr_db):
	signal_power = np.mean(signal**2)
	noise_power = np.mean(noise**2)
	noise_scaling_factor = np.sqrt(signal_power / (noise_power*10**(target_snr_db/10)))
	scaled_noise = noise * noise_scaling_factor
	return signal + scaled_noise, noise_scaling_factor

def compute_snr(measured_signal, target_signal):
	noise = measured_signal - target_signal
	signal_power = np.mean(target_signal ** 2)
	noise_power = np.mean(noise ** 2)
	snr_db = 10 * np.log10(signal_power / noise_power)
	return snr_db
```

---

## 2. Excitation Generation

Generate a cyclostationary excitation signal with random jitter:
```python
rng = default_rng(seed=42)
data_size = 100_000
excitation = np.zeros(data_size)
idx = 0
next_local_idx = 0
f_exitation = 0.005
jitter_amount = 0.05
while idx < data_size:
	excitation[next_local_idx] = 1.0
	next_local_idx = idx + int((1 + rng.normal(size=1, loc=0, scale=jitter_amount)[0]) / f_exitation)
	idx += int(1 / f_exitation)
```

---

## 3. SDOF System Response Simulation

Simulate the response of a single degree-of-freedom system to the excitation:
```python
def generate_sig(f_nat, excitation, fs, zita=0.05):
	amplitude = 1.0
	omega_n = 2 * np.pi * f_nat
	omega_d = omega_n * np.sqrt(1 - zita**2)
	t = np.arange(0, excitation.shape[0] // 2) / fs
	sdof_response = amplitude / omega_d * ((-zita * omega_n) * (-zita * omega_n) * np.exp(-zita * omega_n * t) * np.sin(omega_d * t) +
		(-zita * omega_n) * np.exp(-zita * omega_n * t) * np.cos(omega_d * t) * omega_d +
		(-zita * omega_n) * np.exp(-zita * omega_n * t) * np.cos(omega_d * t) * omega_d +
		np.exp(-zita * omega_n * t) * omega_d * omega_d * (-1) * np.sin(omega_d * t))
	sig = convolve(excitation, sdof_response, mode='full')[:len(excitation)]
	return sig

target = generate_sig(f_nat=0.3, excitation=excitation, fs=1.0)
```

---

## 4. Noise Addition and SNR Control

Add noise to the target signal and control the SNR:
```python
noise = rng.normal(size=data_size)
signal, noise_scale = scale_snr(target, noise, snr=-3)
noise = noise * noise_scale
print(f'snr before filtering {compute_snr(signal, target)}')
```
---

## 5. Cyclostationary Spectral Analysis

Compute and plot the cyclic spectral correlation (CSC) to identify cycle frequencies:
```python
import cyclotorch
import plotly.express as px

alpha, f, S = cyclotorch.CSC_estimators.Faster_SC(
	signal,
	alpha_max=0.026,
	window_len=64,
	convention='asymmetric_negative',
	fs=1.0
)
fig_csc = px.imshow(np.abs(S)[:, f >= 0].T,
					y=f[f >= 0],
					x=alpha,
					aspect="auto", origin='lower', color_continuous_scale='turbo')
fig_csc.update_layout(title='CSC of a sine modulated signal',
					  xaxis_title='cycle frequency',
					  yaxis_title='spectral frequency',
					  template='plotly_white')
fig_csc.show()
```
This plot helps visualize the cyclostationary properties of the signal and identify relevant cycle frequencies for filtering.

---

## 6. FRESH Filter Optimization and Application (with Target)

Compute and apply the optimum and adaptive FRESH filters using the known target signal:
```python
cycle_freq = np.array([i * f_exitation for i in range(-5, 6)])
freshfilt = tc.FRESHfilt.optimum_freshfilt(signal, target, cycle_freq=cycle_freq, filt_len=1024, fs=1.0, impose_hermitian=True)
extracted_sig_optfresh = tc.FRESHfilt.apply_freshfilt(signal, freshfilt).data.numpy()
print(f'obtained snr after filtering using opt freshfilt {compute_snr(extracted_sig_optfresh, target)}')
freshfilt = tc.FRESHfilt.adaptive_freshfilt(signal, target, freshfilt, verbose=True)
extracted_sig_adaptfresh = tc.FRESHfilt.apply_freshfilt(signal, freshfilt).data.numpy()
print(f'obtained snr after filtering using adaptive freshfilt {compute_snr(extracted_sig_adaptfresh, target)}')
```

---

## 7. FRESH Filter Optimization and Application (without Target)

If the target signal is unknown, apply FRESH filtering using only the expected cycle frequencies:
```python
cycle_freq = np.array([i * f_exitation for i in range(-5, 6)])
cycle_freq = cycle_freq[cycle_freq != 0]
freshfilt = tc.FRESHfilt.optimum_freshfilt(signal, signal, cycle_freq=cycle_freq, filt_len=1024, fs=1.0, impose_hermitian=True)
extracted_sig_optfresh = tc.FRESHfilt.apply_freshfilt(signal, freshfilt).data.numpy()
print(f'obtained snr after filtering using opt freshfilt without knowing target signal {compute_snr(extracted_sig_optfresh, target)}')
freshfilt = tc.FRESHfilt.adaptive_freshfilt(signal, target, freshfilt, verbose=True)
extracted_sig_adaptfresh = tc.FRESHfilt.apply_freshfilt(signal, freshfilt).data.numpy()
print(f'obtained snr after filtering using adaptive freshfilt without knowing target signal {compute_snr(extracted_sig_adaptfresh, target)}')
```
For more information see: [Vaerenberg, Rik, et al. "Cyclic filtering for EMI removal and a high central frequency resolution Kurtogram for rolling element bearing diagnostics." Mechanical Systems and Signal Processing 233 (2025): 112716.](https://doi.org/10.1016/j.ymssp.2025.112716)

---
