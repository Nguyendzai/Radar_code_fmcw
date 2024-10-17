import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Radar and signal parameters
fc = 60  # frequency
B = 3239  # sweep bandwidth
S = 54  # sweep slope
fs = 4000  # sampling frequency
Nc = 255  # num chirps in 1 frame
Ns = 128  # num samples of 1 chirp
c0 = 3e+8  # speed of light
R_res = c0 / (2 * B * 1e6)  # range resolution
R_max = fs * c0 / (2 * S * 1e9)  # maximum range

# File reading parameters
file_name = 'Chi_ngang_3m.bin'
num_adc_samples = 128  # number of ADC samples per chirp
num_adc_bits = 16  # number of ADC bits per sample
num_rx = 4  # number of receivers
num_lanes = 2  # number of lanes is always 2
is_real = 0  # set to 1 if real only data, 0 if complex data

# Read file
with open(file_name, 'rb') as fid:
    adc_data = np.fromfile(fid, dtype=np.int16)

# Compensate for sign extension if needed
if num_adc_bits != 16:
    l_max = 2 ** (num_adc_bits - 1) - 1
    adc_data[adc_data > l_max] -= 2 ** num_adc_bits

file_size = adc_data.size
print(file_size)

# For complex data
num_chirps = file_size // (2 * num_adc_samples * num_rx)
file_size = num_chirps * (2 * num_adc_samples * num_rx)

lvds = np.zeros((file_size // 2,), dtype=complex)
lvds[0::2] = adc_data[0:file_size:4] + 1j * adc_data[2:file_size:4]
lvds[1::2] = adc_data[1:file_size:4] + 1j * adc_data[3:file_size:4]
lvds = lvds.reshape((num_chirps, num_adc_samples * num_rx))

new_adc_data = np.zeros((num_rx, num_chirps * num_adc_samples), dtype=np.complex128)
for row in range(num_rx):
    for i in range(num_chirps):
        new_adc_data[row, i * num_adc_samples:(i + 1) * num_adc_samples] = lvds[i, row * num_adc_samples:(row + 1) * num_adc_samples]

def read_rx(rx1):
    rx1 = np.reshape(rx1, (num_chirps, num_adc_samples))
    rx1 = rx1.T
    return rx1

rx1 = read_rx(new_adc_data[0, :])
rx2 = read_rx(new_adc_data[1, :])
rx3 = read_rx(new_adc_data[2, :])
rx4 = read_rx(new_adc_data[3, :])
rx_arr = np.stack((rx1, rx2, rx3, rx4))

# Perform Range FFT
range_fft = np.fft.fft(rx1, axis=0)

# High-pass filter (optional)
b, a = signal.butter(4, 0.01, 'highpass')
range_fft_filtered = signal.lfilter(b, a, range_fft, axis=1)

# Perform Doppler FFT
doppler_fft = np.fft.fft(range_fft_filtered, axis=1)

# Compute the magnitude of the Doppler FFT
doppler_magnitude = np.abs(doppler_fft)

# Plot the Range-Doppler Map
plt.figure(figsize=(12, 8))
plt.imshow(20 * np.log10(doppler_magnitude), aspect='auto', cmap='jet', extent=[0, Nc, 0, R_max])
plt.title('Range-Doppler Map')
plt.ylabel('Range (m)')
plt.xlabel('Doppler Bin')
plt.colorbar(label='Magnitude (dB)')
plt.show()