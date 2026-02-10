# FRESHfilt

Implementation of the FRequency SHifted (FRESH) filters as developped by W.A. Gardner.

### References
- Gardner, William A. Statistical Spectral Analysis: A Nonprobabilistic Theory. Prentice-Hall, Inc., 1986.
- [Gardner, William A. "Cyclic Wiener filtering: theory and method." IEEE Transactions on communications 41.1 (2002): 151-163](https://doi.org/10.1109/26.212375)

--- 

## Optimum FRESH Filter

Compute the optimum FRESH (Frequency Shift) filter for a given signal and target signal.

### Parameters

**signal** : array-like or torch.Tensor

Real input signal to be filtered. Should be 1D.

**target** : array-like or torch.Tensor

Real target signal for filter optimization. Should be 1D and of the same length as `signal`.

**cycle_freq** : array-like or torch.Tensor

Array of cycle frequencies (in Hz) to use for cyclostationary analysis.

**filt_len** : int

Length of the filter.

**fs** : float

Sampling frequency of the signals (in Hz).

**impose_hermitian** : bool, *optional*

If True, enforces Hermitian symmetry on the cycle frequencies and the resulting filter. Note: `cycle_freq` should be symmetric around zero if this is set to True.

**device** : torch.device or str, *optional*

Device to perform computations on (e.g., 'cpu', 'cuda'). If None, automatically selected.

**per_batch** : int, *optional*

Number of batches for computation. Higher values increase speed but use more memory.

### Returns

**FRESHFilter**

An instance of FRESHFilter containing the computed filter coefficients and associated parameters.

### Raises

- **ValueError**: If impose Hermitian is true and the cycle frequencies are not symmetric around zero.

### Notes

- The function uses cyclostationary cross-correlation and auto-correlation matrices for filter estimation.

--- 



## Adaptive FRESH Filter Optimization

Adaptively optimizes the FRESH filter coefficients to minimize the mean squared error between the filtered signal and a target signal.

### Parameters

**signal** : torch.Tensor or array-like

Input signal to be filtered.

**target** : torch.Tensor or array-like

Target signal to match after filtering.

**freshfilt** : FRESHFilter

Instance of FRESHFilter containing initial filterbank and cycle frequencies.

**max_iter** : int, *optional*

Maximum number of optimization iterations (default: 300).

**verbose** : bool, *optional*

If True, prints optimization progress and timing information (default: False).

**device** : torch.device or str, *optional*

Device on which to perform computations. If None, uses default device.

### Returns

**FRESHFilter**

A new FRESHFilter instance with optimized filterbank coefficients.

### Notes

- The function uses the LBFGS optimizer to adapt the real and imaginary parts of the filterbank coefficients.


---


## Apply FRESH Filter

Apply a FRESH filter to a real 1D input signal.

### Parameters

**signal** : torch.Tensor

1D input signal to be filtered.

**freshfilt** : FRESHFilter

FRESHFilter object containing filterbank, cycle frequencies, and sampling frequency.

**device** : torch.device or str, *optional*

Device on which to perform computation (e.g., 'cpu', 'cuda'). If None, uses default device.

### Returns

**torch.Tensor**

The real filtered signal as a 1D tensor on CPU.

### Notes

- There is shifting applied to the filterbank in order to approach zero-phase filtering.
- For short filters (length < 10), uses `torchaudio.functional.convolve`; otherwise, uses FFT-based convolution.
- The output is always moved to CPU and contains only the real part of the filtered signal.

