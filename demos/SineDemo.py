import numpy as np
from numpy.random import default_rng
from scipy.signal import welch,convolve
import plotly.graph_objects as go
import cyclotorch 
from scipy.signal import sosfiltfilt,butter
import plotly.express as px
import torch

    
def scale_snr(signal, noise, target_snr_db):
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    noise_scaling_factor = np.sqrt(signal_power / (noise_power*10**(target_snr_db/10)))
    scaled_noise = noise * noise_scaling_factor
    return signal + scaled_noise,noise_scaling_factor
    
def apply_butterbandpass(arr,lowcut,highcut,fs,order=3):
    sos_bandpassfilter = butter(N=order,Wn=[lowcut,highcut],fs=fs,output='sos',btype='bandpass')
    arr_filtered = sosfiltfilt(sos_bandpassfilter,arr)
    return arr_filtered

def compute_snr(measured_signal, target_signal):
    noise = measured_signal - target_signal
    signal_power = np.mean(target_signal ** 2)
    noise_power = np.mean(noise ** 2)
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db

rng = default_rng(seed=42)

def main():
    data_size =100_000
    snr = 5
    f_envelope = 0.01
    t = np.arange(data_size)
    envelope = np.sin(t*2*np.pi*f_envelope)+1

    # Compute the Power Spectral Density (PSD) of the excitation
    frequencies, psd = welch(envelope, fs=1.0, nperseg=5000)
    
    # Plot the PSD of the exitation
    fig_psd = go.Figure()
    fig_psd.add_trace(go.Scatter(x=frequencies, y=psd, mode='lines', name='PSD'))
    fig_psd.update_layout(title='Power Spectral Density of Sine Envelope',
                          xaxis_title='Frequency (Hz)',
                          yaxis_title='Power Spectral Density',
                          template='plotly_white')
    fig_psd.show()
    
    # Compute the target signal
    target = apply_butterbandpass(rng.normal(size =data_size),lowcut=0.2,highcut=0.3,fs=1.0)*envelope
    
    noise = rng.normal(size =data_size)

    signal,noise_scale = scale_snr(target,noise,snr)
    noise = noise*noise_scale
    signal,target = signal.astype(np.float32),target.astype(np.float32)
    print(f'SNR before filtering {compute_snr(signal,target)}')
    fig_decomp = go.Figure()
    fig_decomp.add_trace(go.Scatter(x=np.arange(len(target))[:10_000], y=signal[:10_000], mode='lines', name='Measured Signal'))
    fig_decomp.add_trace(go.Scatter(x=np.arange(len(target))[:10_000], y=target[:10_000], mode='lines', name='Target Signal'))
    fig_decomp.add_trace(go.Scatter(x=np.arange(len(target))[:10_000], y=noise[:10_000], mode='lines', name='Noise'))
    fig_decomp.update_layout(title='Signal decomposition',
                            xaxis_title='Sample',
                            yaxis_title='Amplitude',
                            template='plotly_white')
    fig_decomp.show()
    
    # Lets plot the SCD
    alpha_arr,f,scd = cyclotorch.CSC_estimators.Faster_SC(
        signal,
        alpha_max=0.05,
        fs=1.0,
        window_len=256,
    )
    alpha_arr,f,scd = alpha_arr.numpy(),f.numpy(),scd.numpy()
    fig_scd = px.imshow(np.abs(scd)[:,f>=0].T ,
                        y= f[f>=0],
                        x=alpha_arr,
                        aspect="auto",origin='lower',color_continuous_scale='turbo')
    fig_scd.update_layout(title='SCD of a sine modulated signal',
                            xaxis_title='cycle frequency',
                            yaxis_title='spectral frequency',
                            template='plotly_white')    
    fig_scd.show()
   
    # And the coherence
    alpha_arr,f,coh = cyclotorch.CSC_estimators.Faster_SC(
        signal,
        alpha_max=0.05,
        fs=1.0,
        window_len=256,
        coherence=True
    )
    alpha_arr,f,coh = alpha_arr.numpy(),f.numpy(),coh.numpy()
    fig_coh = px.imshow(np.abs(coh)[:,f>=0].T ,
                        y= f[f>=0],
                        x=alpha_arr,
                        aspect="auto",origin='lower',color_continuous_scale='turbo')
    fig_coh.update_layout(title='CSCoh of a sine modulated signal',
                            xaxis_title='cycle frequency',
                            yaxis_title='spectral frequency',
                            template='plotly_white')    
    fig_coh.show()
    
    
    
    
    cycle_freq = np.array([-f_envelope,0,f_envelope]).astype(np.float32),
    freshfilt = cyclotorch.FRESHfilt.optimum_freshfilt(signal,target,cycle_freq=cycle_freq,filt_len=1024,fs=1.0,impose_hermitian=True)
    extracted_sig_optfresh = cyclotorch.FRESHfilt.apply_freshfilt(signal,freshfilt).data.numpy()
    print(f'SNR after FRESH filtering {compute_snr(extracted_sig_optfresh,target)}')
    freshfilt = cyclotorch.FRESHfilt.adaptive_freshfilt(signal,target,freshfilt,verbose=True) 
    extracted_sig_adaptfresh = cyclotorch.FRESHfilt.apply_freshfilt(signal,freshfilt).data.numpy()
    print(f'obtained snr after filtering using adaptive freshfilt {compute_snr(extracted_sig_adaptfresh,target)}')
    
    fig_freshfilt = go.Figure()
    fig_freshfilt.add_trace(go.Scatter(x=np.arange(len(target)), y=signal, mode='lines', name='Measured Signal'))
    fig_freshfilt.add_trace(go.Scatter(x=np.arange(len(target)), y=target, mode='lines', name='Target Signal'))
    fig_freshfilt.add_trace(go.Scatter(x=np.arange(len(target)), y=extracted_sig_optfresh, mode='lines', name='Extracted signal'))
    fig_freshfilt.update_layout(title='FRESH Filtered signal',
                            xaxis_title='Sample',
                            yaxis_title='Amplitude',
                            template='plotly_white')
    fig_freshfilt.show()
    
    