import numpy as np
import matplotlib.pyplot as plt

# Tạo chuỗi số phức
data = np.array([np.cos(0.5) + 1j*np.sin(0.5), np.cos(2) +1j*np.sin(2), 
                 np.cos(1.5) +1j*np.sin(1.5), np.cos(3)+ 1j*np.sin(3), 
                 np.cos(2.5) +1j*np.sin(2.5), np.cos(4) +1j*np.sin(4), 
                 np.cos(3.5)+1j*np.sin(3.5), np.cos(5)+1j*np.sin(5)])

# Biến đổi FFT
fft_result = np.fft.fft(data)
fs= 0.8/(2*np.pi)
# Vẽ biểu đồ biên độ phổ
freqs = np.fft.fftfreq(len(data), d= 1/fs)
plt.figure(figsize=(10, 6))
plt.stem(freqs, np.abs(fft_result))
plt.title('Biên độ phổ của tín hiệu sau khi biến đổi FFT')
plt.xlabel('Tần số (normalized)')
plt.ylabel('Biên độ')
plt.grid(True)
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # Thông số
# fs = 1/3  # Tần số lấy mẫu (Hz)
# N = 8    # Số điểm
# t = np.arange(N) / fs  # Thời gian cho 8 điểm đầu tiên

# # Tạo tín hiệu
# signal = np.cos(1/3 * np.pi * t + np.pi / 3) + 1j * np.sin(1/3 * np.pi * t + np.pi / 3)

# # Biến đổi FFT
# fft_result = np.fft.fft(signal)

# # Trục tần số
# freqs = np.fft.fftfreq(N, 1/fs)

# # Vẽ biểu đồ biên độ phổ
# plt.figure(figsize=(14, 6))

# # Biên độ phổ
# plt.subplot(1, 2, 1)
# plt.stem(freqs, np.abs(fft_result))
# plt.title('Biên độ phổ của tín hiệu sau khi biến đổi FFT')
# plt.xlabel('Tần số (Hz)')
# plt.ylabel('Biên độ')
# plt.grid(True)

# # Góc pha
# plt.subplot(1, 2, 2)
# plt.stem(freqs, np.angle(fft_result))
# plt.title('Góc pha của tín hiệu sau khi biến đổi FFT')
# plt.xlabel('Tần số (Hz)')
# plt.ylabel('Góc pha (radians)')
# plt.grid(True)

# plt.show()


