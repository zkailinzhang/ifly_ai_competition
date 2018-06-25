# encoding=utf8


import tensorflow 
import numpy as np 
import os
import librosa
import librosa.display
from random import shuffle
import struct
from pandas import Series,DataFrame
import pandas

import wave  # cont read pcm
import soundfile as sf   #pcm': File contains data in an unknown format.


import matplotlib.pyplot as plt
path = 'F:/1zklcode/python/speaker-recog/kaggle/dataset/'




#show special one .pcm audio file  spectral
onepathwav = 'F:/1zklcode/python/speaker-recog/kaggle/mono_10khz.wav'
onepath  = 'F:/1zklcode/python/speaker-recog/kaggle/dataset/CHANGSHA/changsha/\
dev/speaker31/long/changsha_dev_speaker31_001.pcm'
# wavfile = wave.open(onepath,'rb')
# print(type(wavfile))
#random show one in all path,different direction
#data, samplerate = sf.read(onepath, dtype='float16')


#select one randomly
def function():
	pass


#show spectral
file1 =[]
with open(onepathwav,'rb') as fs:

	filebytes = fs.read() #16bit 2byte  # 逐个记录 每完成
	if not filebytes: 
		pass
	#print(filebytes,type(filebytes))
	#print("\n")
	#file1 = struct.unpack("f", filebytes)[0] error
	#ok np.int16 .float16
	file1 = np.fromstring(str(filebytes),dtype=np.float16)#fs.read()
	#file1 = np.frombuffer(filebytes,dtype=np.int16) #fs.read(4)
	
	print(len(file1))
	print(str(file1)+"\n")
sr =16000
plt.figure()# figsize=(20,10)
plt.subplot(3,1,1)
file1 =np.nan_to_num(file1)

# file1 -= np.mean(file1,axis =0)
# file1/=np.std(file1,axis =0)

s = pandas.Series(file1)
print(s.describe())


print(np.isfinite(file1),np.isfinite(file1).all()) 
#file1 = np.nan_to_num(file1)
print(file1.shape)
# D= librosa.stft(file1)
# print(D.shape)
# DB = librosa.amplitude_to_db(librosa.magphase(D)[0])
# print(DB.shape)
# librosa.display.specshow(DB,x_axis = 'time')
# plt.colorbar(format='%+2.0f dB')
# plt.title('stft')

# ax = plt.subplot(3,1,2)
# S = librosa.feature.melspectrogram(y=file1, sr=sr)
# librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
#                          x_axis='time', y_axis='mel')
# plt.title('melspectrogram')


plt.subplot(3,1,3)
mfccs = librosa.feature.mfcc(y=file1, sr=sr, n_mfcc=40)
print(mfccs.shape)
librosa.display.specshow(mfccs, x_axis='time') #,lable = 'mfcc'
plt.colorbar()
plt.title('mfcc')
# plt.xlim(0, 10)
plt.show() 


# result
# 480597
# [  2.88391113e-02   3.57120000e+04   1.12400000e+03 ...,   2.69750000e+02
#    1.65600000e+03   2.89001465e-02]

# [ True  True  True ...,  True  True  True] True
# (480597,)
# (1025, 939)
# (1025, 939)
# (128, 939)
# [Finished in 28.2s]