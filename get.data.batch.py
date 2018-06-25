#-*- coding: utf-8 -*-
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


path = 'F:/1zklcode/python/speaker-recog/kaggle/dataset/data.dev'

#select one batch randomly

DIALECT = {

    'CHANGSHA':1,
    'HEBEI':2,
    'KEJIA':3,
    'MINNAN':4,
    'NANCHANG':5,
    'SHANGHAI':6

}


# maybe_download(source, DATA_DIR)
batch_features = []
labels = []
sr =16000
batch_size=20
for i in range(batch_size):
    for root,dirs,files in os.walk(path):
        # print(dirs[0])
        # #if not files[0].endswith(".pcm"): continue
        # # print("loaded batch of %d files" % len(files))
        shuffle(dirs)
        shuffle(files)
        if not files :continue
        if not files[0].endswith(".pcm"): continue
        root = root.replace("\\","/")
        #print(root,files)
        # allpath = []
        # for file in files:
        #     print(root, file)
        #     print(os.path.join(root,file )  )
        # for allpath1 in allpath:
        #     print(allpath1 )

        #root = root.replace("\\", "/")
        #for file in files:
        splitlist = (os.path.join(root,files[0])).split('/')
        # print(splitlist,splitlist[7])
        lab = DIALECT.get(splitlist[7])
        #  print(lab)
        #if not lab :continue
        for file in files:
            if not file.endswith(".pcm"): continue
            datapcm =[]
            with open(os.path.join(root,file),'rb') as fs:
                data = fs.read()
                if not data: pass
                #data =   # what 二进制？   转字符串
                datapcm = np.frombuffer(data,dtype=np.int16)
                #datapcm = np.fromstring(str(data), dtype=np.int16)  #偶然ValueError: string size must be a multiple of element size
            datapcm = [float(dataint) for dataint in datapcm]
            mfcc = librosa.feature.mfcc(np.array(datapcm), sr=sr,n_mfcc =13)
            batch_features.append(mfcc)
            labels.append(lab)
            print(file,mfcc.shape,lab)




            break
        break




# "D:\Program Files\Anaconda3\python.exe" F:/1zklcode/python/speaker-recog/kaggle/get.data.batch.py
# minnan_train_speaker30_036.pcm (13, 120) 4
# minnan_train_speaker23_086.pcm (13, 149) 4
# hebei_train_speaker23_078.pcm (13, 202) 2
# changsha_train_speaker16_125.pcm (13, 76) 1
# minnan_train_speaker15_068.pcm (13, 369) 4
# changsha_train_speaker13_091.pcm (13, 194) 1
# minnan_train_speaker24_028.pcm (13, 127) 4
# nanchang_train_speaker24_059.pcm (13, 208) 5
# shanghai_train_speaker01_148.pcm (13, 112) 6
# minnan_train_speaker13_181.pcm (13, 77) 4
# hebei_train_speaker07_066.pcm (13, 140) 2
# changsha_train_speaker16_004.pcm (13, 282) 1
# changsha_train_speaker29_004.pcm (13, 163) 1
# hebei_train_speaker23_157.pcm (13, 77) 2
# nanchang_train_speaker21_161.pcm (13, 136) 5
# kejia_train_speaker24_107.pcm (13, 72) 3
# kejia_train_speaker25_088.pcm (13, 123) 3
# hebei_train_speaker23_130.pcm (13, 74) 2
# nanchang_train_speaker22_109.pcm (13, 162) 5
# nanchang_train_speaker28_111.pcm (13, 83) 5
#
# Process finished with exit code 0