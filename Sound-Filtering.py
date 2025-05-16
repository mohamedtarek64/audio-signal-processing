import numpy as np #m
import scipy as sp #f
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from scipy import signal #b
import matplotlib.pyplot as plt

# تحديد مسار الملف الصوتي
input_path = "eagle.wav"
output_path = "New_Sound_With_Bandpass_Filter_And_Scaled.wav"

# قراءة الملف الصوتي
(Frequency, array) = read(input_path)

# عرض طول المصفوفة
print("Length of the array:", len(array))

# تحويل فورييه للإشارة باستخدام دالة fft من scipy.fft
FourierTransformation = sp.fft.fft(array)

# حساب الترددات بشكل صحيح باستخدام np.fft.fftfreq
scale = np.fft.fftfreq(len(array), d=1/Frequency)  # حساب الترددات

# إعداد الشكل لعرض الرسوم البيانية في شبكة 2x2
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# رسم الإشارة الأصلية
axes[0, 0].plot(array)
axes[0, 0].set_title('Original Signal Spectrum')
axes[0, 0].set_xlabel('Samples')
axes[0, 0].set_ylabel('Amplitude')

# رسم طيف الإشارة بعد تطبيق FFT
axes[0, 1].stem(scale[:len(scale)//2], np.abs(FourierTransformation[:len(FourierTransformation)//2]), 'r')  # عرض النصف الأول من الطيف
axes[0, 1].set_title('Signal spectrum after FFT')
axes[0, 1].set_xlabel('Frequency (Hz)')
axes[0, 1].set_ylabel('Amplitude')

# تطبيق فلتر Bandpass
low_cut = 500  # تردد القطع السفلي بالـ Hz
high_cut = 3000  # تردد القطع العلوي بالـ Hz
b, a = signal.butter(5, [low_cut/(Frequency/2), high_cut/(Frequency/2)], btype='bandpass')  # فلتر Bandpass
filteredSignal = signal.lfilter(b, a, array)

# تقليل السعة (نصف السعة في هذه الحالة)
scaledSignal = filteredSignal * 0.5  # ضرب الإشارة في 0.5 لتقليل السعة

# رسم الإشارة بعد تطبيق فلتر Bandpass وتقليل السعة
axes[1, 0].plot(scaledSignal)
axes[1, 0].set_title('Bandpass Filter (500 Hz - 3000 Hz) and Scaled')
axes[1, 0].set_xlabel('Samples')
axes[1, 0].set_ylabel('Amplitude')

# عرض جميع الرسوم البيانية
plt.tight_layout()
plt.show()

# حفظ الملف الصوتي المعدل بعد تطبيق الفلتر وتقليل السعة
write(output_path, Frequency, scaledSignal)
