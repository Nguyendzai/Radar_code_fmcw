from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

# Thiết kế bộ lọc thông cao số (digital high-pass filter)
b, a = signal.butter(8, 0.075, 'high', analog=False) 

# Tính toán đáp ứng tần số của bộ lọc số
w, h = signal.freqz(b, a, worN=2000)
frequencies = w * 1000 / (2 * np.pi)  # Chuyển đổi sang tần số thực tế (Hz)

# Vẽ đáp ứng tần số
plt.plot(frequencies, 20 * np.log10(abs(h)))
plt.xlim([0, 100])  # Đặt giới hạn trục x từ 0 đến 10,000 Hz
plt.title('Butterworth Filter Frequency Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
# plt.axvline(1000, color='green')  # Tần số cắt (cutoff frequency)
plt.show()


# t = np.linspace(0, 1, 1000, False)  # 1 second
# sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# ax1.plot(t, sig)
# ax1.set_title('10 Hz and 20 Hz sinusoids')
# ax1.axis([0, 1, -2, 2])

# sos = signal.butter(10, 15, 'hp', fs=1000, output='sos')
# filtered = signal.sosfilt(sos, sig)
# ax2.plot(t, filtered)
# ax2.set_title('After 15 Hz high-pass filter')
# ax2.axis([0, 1, -2, 2])
# ax2.set_xlabel('Time [seconds]')
# plt.tight_layout()
# plt.show()

# from scipy import signal
# import matplotlib.pyplot as plt
# import numpy as np

# # Tần số cắt chuẩn hóa
# normalized_cutoff = 0.0075

# # Thiết kế bộ lọc Butterworth thông cao
# b, a = signal.butter(4, normalized_cutoff, btype='high')

# # Tính đáp ứng tần số của bộ lọc tương tự
# w, h = signal.freqs(b, a)

# # Vẽ đáp ứng tần số
# plt.semilogx(w, 20 * np.log10(abs(h)))
# plt.title('Butterworth filter frequency response')
# plt.xlabel('Frequency [radians / second]')
# plt.ylabel('Amplitude [dB]')
# plt.margins(0, 0.1)
# plt.grid(which='both', axis='both')

# # Hiển thị tần số cắt chuẩn hóa
# cutoff_freq = normalized_cutoff * np.pi  # Tần số cắt thực tế (rad/s)
# plt.axvline(cutoff_freq, color='green', linestyle='--', label='Cutoff frequency')

# plt.legend()
# plt.show()

# NTS = 10
# nc = 1
# Data_range= np.zeros((int(NTS/2),nc))

# tmp = np.ones((100, 1))

# Data_range = tmp[int(NTS/2):, :]
# n = np.size(Data_range,1)
# print(n)
# a = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
# a = np.reshape(a, (3, 4))
# print(a)