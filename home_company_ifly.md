## 2018.6.10
1. python from import: package.model.function;package.model.attribution
2. 采样率 16khz，600ms的音频，节点是16*600 个，
3. 量化 16bit 采样得到的数据只是一些离散值，这些离散值应该能用计算机中的若干二进制的
位来表示。采样频率44.1kHz，量化位数16位，意味着每秒采集数据44.1k个，每个数据占2字节，这是一个声道的数据，双声道再乘以2，最后结果再乘以60秒，就是44.1×1000×2×2×60=10584000字节，1MB=1024×1024=1048576字节，所以一分钟的存储容量为10584000/1048576=10.09MB，约为10.1MB
4. 分帧 帧长 帧位移  wlen inc 时域分帧10-30ms，
5. 加窗和分帧都是语音信号提取特征的预处理阶段。先分帧，后加窗，再做快速傅里叶变换。
分帧：简单来说，一段语音信号整体上看不是平稳的，但是在局部上可以看作是平稳的。在后期的语音处理中需要输入的是平稳信号，所以要对整段语音信号分帧，也就是切分成很多段。在10-30ms范围内都可以认为信号是稳定的，一般以不少于20ms为一帧，1/2左右时长为帧移分帧。帧移是相邻两帧间的重叠区域，是为了避免相邻两帧的变化过大。
加窗：按上述方法加窗后，每一帧的起始段和末尾端会出现不连续的地方，所以分帧越多与原始信号的误差也就越大。加窗就是为了解决这个问题，使分帧后的信号变得连续，每一帧就会表现出周期函数的特征。在语音信号处理中一般加汉明窗。
6. python io read 若二进制 怎是字节，若文本则字符串
7. WAV format	Min	Max	NumPy dtype
32-bit floating-point	-1.0	+1.0	float32
32-bit PCM	-2147483648	+2147483647	int32
16-bit PCM	-32768	+32767	int16
8-bit PCM	0	255	uint8

8. Converting wav file 16-bit PCM data into float or intege
9. 
9. librosa.stft  时频特性
9. librosa.feature.melspectrogram   

11. librosa.feature.mfcc 

10. librosa.feature.specshow 把上面显示
12. python的struct模块
13. 这里我们的数据类型是float32型的，对应过来是4bytes，使用for循环逐个read4个字节。
14. 帧 就是时间