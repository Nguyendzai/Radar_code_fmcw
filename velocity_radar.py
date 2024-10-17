import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
fc = 60      # frequency
B = 3239     # sweep bandwidth
S = 54       # sweep slope
fs = 4000    # sampling freq
Nc = 255     # num chirps in 1 frame
Ns = 128     # num samples of 1 chirp
c0 = 3e+8
R_res = c0 / (2e+6*B)       # range resolution
R_max = fs*c0 / (2e+9*S)
file_name = r'V_0.2_ms.bin'
# global variables
num_adc_samples = 128  # number of ADC samples per chirp
num_adc_bits = 16  # number of ADC bits per sample
num_rx = 4  # number of receivers
num_lanes = 2  # do not change. number of lanes is always 2
is_real = 0  # set to 1 if real only data, 0 if complex data
# read file
with open(file_name, 'rb') as fid:
    adc_data = np.fromfile(fid, dtype=np.int16)

# if 12 or 14 bits ADC per sample compensate for sign extension
if num_adc_bits != 16:
    l_max = 2**(num_adc_bits-1)-1
    adc_data[adc_data > l_max] -= 2**num_adc_bits

file_size = adc_data.size
print(file_size)
# for complex data
num_chirps = file_size // (2 * num_adc_samples * num_rx)
file_size = num_chirps * (2 * num_adc_samples * num_rx)

lvds = np.zeros((file_size // 2,), dtype=complex)

lvds[0::2] = adc_data[0:file_size:4] + 1j * adc_data[2:file_size:4]
lvds[1::2] = adc_data[1:file_size:4] + 1j * adc_data[3:file_size:4]

lvds = lvds.reshape((num_chirps, num_adc_samples * num_rx))

new_adc_data = np.zeros((num_rx, num_chirps * num_adc_samples), dtype=np.complex128)
for row in range(num_rx):
    for i in range(num_chirps):
        new_adc_data[row, i*num_adc_samples:(i+1)*num_adc_samples] = lvds[i, row*num_adc_samples:(row+1)*num_adc_samples]


def read_rx(rx1):
    rx1 = np.reshape(rx1, (num_chirps, num_adc_samples))
    rx1 = rx1.T
    return rx1
# rx11 = np.reshape(new_adc_data[0, :].T, (num_chirps, 128))
# print(rx11.shape)
rx1 = read_rx(new_adc_data[0, :])
rx2 = read_rx(new_adc_data[1, :])
rx3 = read_rx(new_adc_data[2, :])
rx4 = read_rx(new_adc_data[3, :])
rx_arr = np.stack((rx1, rx2, rx3, rx4))
# FFT fast time
range_fft = np.fft.fft(rx1, axis=0)
#Filter highpass
b, a = signal.butter(4, 0.0075 , 'highpass')
Data_range_MTI = signal.lfilter(b,a,range_fft, axis=1)
# Perform Doppler FFT
doppler_fft = np.fft.fft(Data_range_MTI, axis=1)
# Compute the magnitude of the Doppler FFT
doppler_magnitude = np.abs(doppler_fft)
# Shift zero-frequency component to center of spectrum
doppler_magnitude = np.fft.fftshift(doppler_magnitude, axes=1)

# Velocity resolution
velocity_res = c0 / (2e+10 * fc * (Nc / fs))
# Calculate the velocity bins
# velocity_bins = np.linspace(-Nc / 2, Nc / 2 - 1, Nc) * velocity_res
velocity_bins = np.fft.fftfreq(Nc, d=1/fs) * velocity_res
velocity_bins = np.fft.fftshift(velocity_bins)
print(velocity_bins.min())
print(velocity_bins.max())
# Draw fig
fig, ax = plt.subplots(figsize=(30, 12))

im = ax.imshow(20 * np.log10(np.abs(doppler_fft)), cmap='jet', aspect='auto', extent=[velocity_bins.min(), velocity_bins.max(), R_max, 0])

# Set the colorbar limits
clim = im.get_clim()
im.set_clim(clim[1] - 70, clim[1])

# Flip the y-axis to match MATLAB's behavior
ax.invert_yaxis()

# Add the colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Magnitude (dB)')

# Show the plot
plt.show()

