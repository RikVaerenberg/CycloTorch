import numpy as np
from numpy.random import default_rng
from scipy.signal import welch,convolve
import plotly.graph_objects as go
import sys
import cyclotorch 
import plotly.express as px

def generate_sig(f_nat,excitation,fs,zita=0.05):
    # See D’Elia, Gianluca, Marco Cocconcelli, and Emiliano Mucchi. "An algorithm for the simulation of faulted bearings in non-stationary conditions." Meccanica 53 (2018): 1147-1166.
    amplitude = 1.0
    omega_n = 2*np.pi*f_nat
    omega_d = omega_n*np.sqrt(1-zita**2)
    t = np.arange(0,excitation.shape[0]//2)/fs
    # system response
    sdof_response = amplitude/omega_d*((-zita*omega_n)*(-zita*omega_n)*np.exp(-zita*omega_n*t)*np.sin(omega_d*t) + \
        (-zita*omega_n)*np.exp(-zita*omega_n*t)*np.cos(omega_d*t)*omega_d + \
            (-zita*omega_n)*np.exp(-zita*omega_n*t)*np.cos(omega_d*t)*omega_d + \
                np.exp(-zita*omega_n*t)*omega_d*omega_d*(-1)*np.sin(omega_d*t))
    sig = convolve(excitation,sdof_response,mode='full')[:len(excitation)]
    return sig
    
def scale_snr(signal, noise, target_snr_db):
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    noise_scaling_factor = np.sqrt(signal_power / (noise_power*10**(target_snr_db/10)))
    scaled_noise = noise * noise_scaling_factor
    return signal + scaled_noise,noise_scaling_factor
    
def compute_snr(measured_signal, target_signal):
    noise = measured_signal - target_signal
    signal_power = np.mean(target_signal ** 2)
    noise_power = np.mean(noise ** 2)
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db

rng = default_rng(seed=42)

def main():
    data_size =100_000
    snr = -3
    
    # Compute the excitation function for a CS model 
    # See Borghesani, Pietro, et al. "Bearing signal models and their effect on bearing diagnostics." Mechanical Systems and Signal Processing 174 (2022): 109077.
    excitation = np.zeros(data_size)
    idx = 0
    next_local_idx = 0
    f_exitation = 0.005
    jitter_amount = 0.05
    while idx<data_size:
        excitation[next_local_idx] = 1.0
        next_local_idx = idx + int((1+rng.normal(size=1,loc=0,scale=jitter_amount)[0])/f_exitation)
        idx += int(1/f_exitation)

    # Compute the Power Spectral Density (PSD) of the excitation
    frequencies, psd = welch(excitation, fs=1.0, nperseg=5000)
    
    # Plot the PSD 
    fig_psd = go.Figure()
    fig_psd.add_trace(go.Scatter(x=frequencies, y=psd, mode='lines', name='PSD'))
    fig_psd.update_layout(title='Power Spectral Density of Excitation',
                          xaxis_title='Frequency (Hz)',
                          yaxis_title='Power Spectral Density',
                          template='plotly_white')
    fig_psd.show()
    # Compute the target signal
    target = generate_sig(f_nat= 0.3,excitation=excitation,fs=1.0)
    
    noise = rng.normal(size =data_size)

    signal,noise_scale = scale_snr(target,noise,snr)
    noise = noise*noise_scale
    print(f'snr before filtering {compute_snr(signal,target)}')
    
    signal,target = signal.astype(np.float32),target.astype(np.float32)

    fig_decomp = go.Figure()
    fig_decomp.add_trace(go.Scatter(x=np.arange(len(target)), y=signal, mode='lines', name='Measured Signal'))
    fig_decomp.add_trace(go.Scatter(x=np.arange(len(target)), y=target, mode='lines', name='Target Signal'))
    fig_decomp.add_trace(go.Scatter(x=np.arange(len(target)), y=noise, mode='lines', name='Noise'))
    fig_decomp.update_layout(title='Signal decomposition',
                            xaxis_title='Sample',
                            yaxis_title='Amplitude',
                            template='plotly_white')
    fig_decomp.show()
    
    
    # Lets check what the CSC looks like to identify the cycle frequencies
    alpha,f,S = cyclotorch.CSC_estimators.Faster_SC(
        signal,
        alpha_max=0.026,
        window_len=64,
        convention='asymmetric_negative',
        fs=1.0
    )   
    fig_csc = px.imshow(np.abs(S)[:,f>=0].T ,
                        y= f[f>=0],
                        x=alpha,
                        aspect="auto",origin='lower',color_continuous_scale='turbo')
    fig_csc.update_layout(title='CSC of a sine modulated signal',
                            xaxis_title='cycle frequency',
                            yaxis_title='spectral frequency',
                            template='plotly_white')    
    fig_csc.show()
    
    
    cycle_freq = np.array([i*f_exitation for i in range(-5,6)]).astype(np.float32)

    freshfilt = cyclotorch.FRESHfilt.optimum_freshfilt(signal,target,cycle_freq=cycle_freq,filt_len=1024,fs=1.0,impose_hermitian=True)
    extracted_sig_optfresh = cyclotorch.FRESHfilt.apply_freshfilt(signal,freshfilt).data.numpy()
    print(f'obtained snr after filtering using opt freshfilt {compute_snr(extracted_sig_optfresh,target)}')
    freshfilt = cyclotorch.FRESHfilt.adaptive_freshfilt(signal,target,freshfilt,verbose=True) 
    extracted_sig_adaptfresh = cyclotorch.FRESHfilt.apply_freshfilt(signal,freshfilt).data.numpy()
    print(f'obtained snr after filtering using adaptive freshfilt {compute_snr(extracted_sig_adaptfresh,target)}')

    fig_freshfilt = go.Figure()
    fig_freshfilt.add_trace(go.Scatter(x=np.arange(len(target)), y=signal, mode='lines', name='Measured Signal'))
    fig_freshfilt.add_trace(go.Scatter(x=np.arange(len(target)), y=target, mode='lines', name='Target Signal'))
    fig_freshfilt.add_trace(go.Scatter(x=np.arange(len(target)), y=extracted_sig_optfresh, mode='lines', name='Extracted signal'))
    fig_freshfilt.update_layout(title='Signal obtained by FRESH filtering based on target signal',
                            xaxis_title='Sample',
                            yaxis_title='Amplitude',
                            template='plotly_white')
    fig_freshfilt.show()
    
    # In case we do not know the target signal, but we know the expected cycle frequencies we can still extract something following 
    # Vaerenberg, Rik, et al. "Cyclic filtering for EMI removal and a high central frequency resolution Kurtogram for rolling element bearing diagnostics." Mechanical Systems and Signal Processing 233 (2025): 112716.
    # But make sure to not include the zero cycle freq
    cycle_freq = np.array([i*f_exitation for i in range(-5,6)]).astype(np.float32)
    cycle_freq = cycle_freq[cycle_freq!=0]
    freshfilt = cyclotorch.FRESHfilt.optimum_freshfilt(signal,signal,cycle_freq=cycle_freq,filt_len=1024,fs=1.0,impose_hermitian=True)
    extracted_sig_optfresh = cyclotorch.FRESHfilt.apply_freshfilt(signal,freshfilt).data.numpy()
    print(f'obtained snr after filtering using opt freshfilt without knowing target signal {compute_snr(extracted_sig_optfresh,target)}')
    freshfilt = cyclotorch.FRESHfilt.adaptive_freshfilt(signal,signal,freshfilt,verbose=True) # Unless there is some numerical stability with the optimum freshfilt, this should not change much
    extracted_sig_adaptfresh = cyclotorch.FRESHfilt.apply_freshfilt(signal,freshfilt).data.numpy()
    print(f'obtained snr after filtering using adaptive freshfilt without knowing target signal {compute_snr(extracted_sig_adaptfresh,target)}')
    
    fig_freshfilt = go.Figure()
    fig_freshfilt.add_trace(go.Scatter(x=np.arange(len(target)), y=signal, mode='lines', name='Measured Signal'))
    fig_freshfilt.add_trace(go.Scatter(x=np.arange(len(target)), y=target, mode='lines', name='Target Signal'))
    fig_freshfilt.add_trace(go.Scatter(x=np.arange(len(target)), y=extracted_sig_optfresh, mode='lines', name='Extracted signal'))
    fig_freshfilt.update_layout(title='Signal obtained by FRESH filter based extraction',
                            xaxis_title='Sample',
                            yaxis_title='Amplitude',
                            template='plotly_white')
    fig_freshfilt.show()