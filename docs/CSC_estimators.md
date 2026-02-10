# CSC_estimators

A set of estimators in order to estimate the (cross) cyclic spectral correlation or coherence

---

## CSC_ACP
Estimate the (Cross) Cyclic Spectral Correlation (CSC) or Coherence using the Average Cyclic Periodogram (ACP) estimator.


### Parameters


**x** : torch.Tensor or array-like

Input signal to analyze. Will be flattened and cast to a torch tensor.

**y** : torch.Tensor or array-like

Second input signal (for cross-spectral analysis). Will be flattened and cast to a torch tensor. For the computation of the auto-CSC, pass the same array as x.

**alpha_arr** : torch.Tensor or array-like

Array of cycle frequencies (Hz) to compute.

**window_len** : int

Length of the window function to use for STFT.

**nfft** : int or None, *optional*

Number of FFT points. If None, defaults to window_len.

**n_overlap** : int or None, *optional*

Number of overlapping samples between windows. If None, hop_len must be set.

**hop_len** : int or None, *optional*

Hop size between windows. If None, n_overlap must be set.

**window** : str, *optional*

Type of window to use. Default is 'hann'.

**fs** : float, *optional*

Sampling frequency of the input signal. Default is 1.0.

**convention** : {'symmetric', 'asymmetric_negative', 'asymmetric_positive'}, *optional*

Which CSC convention to use. Default is 'symmetric'.

**coherence** : bool, *optional*

If True, computes the coherence instead of the correlation. Default is False.

**device** : torch.device or str or None, *optional*

Device to perform computation on. If None, uses cuda device if available.

**results_device** : torch.device, *optional*

Device to store results. Default is CPU. Can be set to GPU for faster computation (requires more memory).

**per_batch** : int, *optional*

Batch size for computation. Lower if out-of-memory errors occur. Default is 20.

### Returns

**out_tensor** : torch.Tensor  

Spectral correlation (or coherence), shape `(len(alpha_arr), nfft)`. This array is still on the given device.

### Raises

- **ValueError**: If `convention` is not one of ['symmetric', 'asymmetric_negative', 'asymmetric_positive'].

### Notes

- Batch processing is used for memory efficiency; adjust `per_batch` for large signals or limited GPU memory.

### Reference
- [Spooner, Chad M., and William A. Gardner. "The cumulant theory of cyclostationary time-series. II. Development and applications." IEEE Transactions on Signal Processing 42.12 (2002): 3409-3429](https://doi.org/10.1109/78.340776)


---


## CSC_FSM

Estimate the (Cross) Cyclic Spectral Correlation (CSC) or Spectral Coherence using the Frequency Smoothing Method (FSM).

### Parameters

**x** : torch.Tensor

Input signal (1D tensor).

**y** : torch.Tensor

Second input signal (1D tensor), must have the same length as `x`. For the computation of the auto-CSC, pass the same array as x.

**alpha_arr** : torch.Tensor

Array of cyclic frequencies (1D tensor).

**smooth_len** : int

Length of the frequency smoothing window.

**fs** : float, *optional*

Sampling frequency of the input signals. Default is 1.0.

**convention** : {'symmetric', 'asymmetric_negative', 'asymmetric_positive'}, *optional*

Which CSC convention to use. Default is 'symmetric'.

**coherence** : bool, *optional*

If True, compute spectral coherence instead of SCD. Default is False.

**device** : torch.device or None, *optional*

Device to perform computation on. If None, automatically selects CUDA if available, otherwise CPU.

**per_batch** : int, *optional*

Number of cyclic frequencies to process per batch for optimization. Lower this value if out-of-memory errors occur. Default is 10.

### Returns

**torch.Tensor**

Cyclic spectral correlation or coherence tensor of shape (len(alpha_arr), len(x)).

### Raises

- **ValueError**: If `convention` is not one of ['symmetric', 'asymmetric_negative', 'asymmetric_positive'].
- **ValueError**: If `x` and `y` do not have the same length.

### Notes

- For large input sizes or cyclic frequency arrays, adjust `per_batch` to avoid memory issues.

### Reference
- [Spooner, Chad M., and William A. Gardner. "The cumulant theory of cyclostationary time-series. II. Development and applications." IEEE Transactions on Signal Processing 42.12 (2002): 3409-3429](https://doi.org/10.1109/78.340776)


---


## Fast_SC

Compute the cyclic spectral correlation using the fast spectral correlation method of J. Antoni.

### Parameters

**x** : torch.Tensor or array-like

Input signal to analyze. Will be flattened and cast to a torch tensor.

**alpha_max** : float

Maximum cycle frequency to compute (Hz).

**window_len** : int

Length of the window function to use for STFT.

**window** : str, *optional*

Type of window to use.

**fs** : float, *optional*

Sampling frequency of the input signal. Default is 1.0.

**R** : int or None, *optional*

Hop size between windows. If None, it is automatically determined to obtain the required alpha_max and have at least 75% overlapping windows.

**coherence** : bool, *optional*

If True, computes the coherence instead of the correlation. Default is False.

**convention** : {'symmetric', 'asymmetric_negative', 'asymmetric_positive'}, *optional*

Which CSC convention to use. Default is 'asymmetric_positive' following the implementation of J.Antoni in Matlab.

**device** : torch.device or str or None, *optional*

Device to perform computation on. If None, uses cuda device if available.

### Returns

**alpha** : torch.Tensor

Array of cycle frequencies (Hz) up to `alpha_max`.

**f** : torch.Tensor

Array of spectral frequencies (Hz), for the symmetric convention the spectral resolution is doubled.

**S** : torch.Tensor

Spectral correlation (or coherence), shape (len(alpha), len(f)).

### Raises

- **ValueError**: If `convention` is not one of ['symmetric', 'asymmetric_negative', 'asymmetric_positive'].
- **ValueError**: If the window is not recognized.

### Reference
- [Antoni, Jérôme, Ge Xin, and Nacer Hamzaoui. "Fast computation of the spectral correlation." Mechanical systems and signal processing 92 (2017): 248-277](https://doi.org/10.1016/j.ymssp.2017.01.011)

---


## SSCA

Compute the (cyclic) spectral correlation using the Strip Spectral Correlation Algorithm (SSCA).

### Parameters

**x** : torch.Tensor or array-like

Input signal to analyze. Will be flattened and cast to a torch tensor.

**window_len** : int

Window length used for the STFT, determines the spectral frequency resolution.

**window** : str, *optional*

Type of window to use. Default is 'hann'.

**fs** : float, *optional*

Sampling frequency of the input signal. Default is 1.0.

**coherence** : bool, *optional*

If True, computes the cyclic coherence instead of the correlation. Default is False.

**convention** : {'asymmetric_negative', 'asymmetric_positive'}, *optional*

Which CSC convention to use. Default is 'asymmetric_negative'.

**device** : torch.device or str or None, *optional*

Device to perform computation on. If None, uses cuda device if available.

### Returns

**alpha** : torch.Tensor

Cycle frequency vector (Hz).

**f** : torch.Tensor

 Spectral frequency vector (Hz), only the positive spectral frequencies are computed.

**S** : torch.Tensor

Spectral correlation (or coherence), shape (len(alpha), len(f)).

### Raises

- **ValueError**: If `convention` is not one of ['asymmetric_negative', 'asymmetric_positive'].

### Reference
- [Roberts, R. S., Brown, W. A., Loomis, H. H., & Helms, J. R. (1991). Computationally efficient algorithms for cyclic spectral analysis. IEEE Signal Processing Magazine, 8(2), 38-49.](https://doi.org/10.1109/79.81008)
- [Borghesani, Pietro, and Jérôme Antoni. "A faster algorithm for the calculation of the fast spectral correlation." Mechanical Systems and Signal Processing 111 (2018): 113-118](https://doi.org/10.1016/j.ymssp.2018.03.059)

---

## Faster_SC

Compute the (cyclic) spectral correlation using the faster spectral correlation method by P. Borghesani.

### Parameters

**x** : torch.Tensor or array-like

Input signal to analyze. Will be flattened and cast to a torch tensor.

**alpha_max** : float

Maximum cycle frequency to compute (Hz).

**window_len** : int

Length of the window function to use for STFT.

**window** : str, *optional*

Type of window to use.

**fs** : float, *optional*

Sampling frequency of the input signal. Default is 1.0.

**R** : int or None, *optional*

Hop size between windows. If None, it is automatically determined to obtain the required alpha_max and have at least 75% overlapping windows.

**coherence** : bool, *optional*

If True, computes the coherence instead of the correlation. Default is False.

**convention** : {'asymmetric_negative', 'asymmetric_positive'}, *optional*

Which CSC convention to use. Default is 'asymmetric_negative'.

**device** : torch.device or str or None, *optional*

Device to perform computation on. If None, uses cuda device if available.

### Returns

**alpha** : torch.Tensor

Array of cycle frequencies (Hz) up to `alpha_max`.

**f** : torch.Tensor

Array of spectral frequencies (Hz).

**S** : torch.Tensor

Spectral correlation (or coherence), shape (len(alpha), len(f)).

### Raises

- **ValueError**: If `convention` is not one of ['asymmetric_negative', 'asymmetric_positive'].
- **ValueError**: If the window is not recognized.

### Notes

- The coherence computation is done based on the prewhitened array instead of the proposed method by P. Borghesani for numerical stability.

### Notes

- For cyclic spectral coherence estimation, additional prewhitening is performed for robustness with respect to the MATLAB code of Jerome Antoni.

### Reference
- [Borghesani, Pietro, and Jérôme Antoni. "A faster algorithm for the calculation of the fast spectral correlation." Mechanical Systems and Signal Processing 111 (2018): 113-118](https://doi.org/10.1016/j.ymssp.2018.03.059)





