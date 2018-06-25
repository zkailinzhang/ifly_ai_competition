##讯飞##

1. 把mp3转换wav 。pcm 用python读，
2. subprocess 调 ffmpeg 或者 mencoder 最方便，，audiotools
3. 音素级标音文件phones0.mlf
4. 两个语音处理工具：HTK和kaldi，MFCC 语音数据特征提取和降低运算维度
5. 1. 用cooledit打开，是不是选项没选对，要选16k 16bit LSB
格式 通道  采样  
sample rate：16000 44100
channel ：mono /stereo
resolution 8bit 16bit 32bit

data formatted as: 16bit intel pcm(lsb,msb);16bit Motorola pcm(lsb,msb);8bit mu-law compressed;8bit A-law compressed;

6. 说话人识别对应物体识别，语音识别成文字 对应物体检测，，方言分类，说话人分两类，对应物体匪类，
7. end to end语音识别系统
8. MFCC，CNN，LSTM，QRNN，CTC，
9. Cover Song ID with Beat-Synchronous Chroma 用节拍同步色度，识别歌曲
10. 采样率 采样带宽（字节）
11. harmonic 柔和的音乐 percussive 节拍音乐
12. librosa pysoundfile ffmpeg speechrognize
13. wav ok
14. ogg ok
15. pcm no ok
16. <font color=red>数据以采样率16000Hz，16比特量化的PCM格式存储。</font>




##爱奇艺##
1. 多模态任务识别：人脸，步态声音，多姿态 多表情 多化妆  多年龄 光照 遮挡
2. 50万条视频，5000个人物 1-30秒时长 **2018/7/1 下载** 
3. 参赛者使用训练好的多模态人物识别模型，预测视频测试集中出现的人物身份，可以使用视觉，语音，字幕等多种模态的识别方法。
4. 测试人物身份识别的Top N结果中的正确比例
5. topN评价方法

##美图短视频分类##
1. 技能分享、幽默搞怪、时尚潮流、社会热点、街头采访等主题
2. 组织方将提供GPU集群环境对参赛团队提交的算法模型进行验证与系统测试，组织方随后将以邮件形式将具体Docker镜像格式通知给参赛方。
3. 
- 系统版本	Ubuntu 16.04 x64
- 编译环境	gcc 4.9.1; boost 1.55;protobuf 2.5.0
- 第三方库	OpenCV 3.4.0; Python 2.7, Numpy 1.11.0
- 	NVIDIA GTX1080Ti
4. docker镜像
5. 评判标准包含“短视频算法准确率”以及“短视频实时分类”两个方面。
6. 共计20类，每类约2000个视频，视频长度约为5 – 15s。**2018/6/20 下载** 
7. 按最高分类置信度输出预测标签l
8. 参赛队伍程序的输入数据为视频文件，输出为该视频的类别。运行时间包括所有推理和预处理时间，如视频截帧、提取音频信息、提取光流信息等。
9. 结果提交方式
	- 参赛团队需要把运行环境、模型与可执行程序打包为Docker镜像文件提交。每支参赛队伍只能提交一份最终模型和可执行代码。
	- Dockerfile该文件包含构建镜像所需要的程序、框架版本号和构建方式。
10. <font color=red>内容行人检测的准确率以mAP为评价指标</font>，可以通过四步计算：P，R，AP，MAP。

Precision(P): 在检测的结果中，正确的检测结果占总检测结果的比例。

Recall(R): 在所有的groundtruth中，被正确检测出来的占总groundtruth数量的比例。

Average Precision(AP): PR曲线下的面积，即对P取平均值。

Mean Average Precision(MAP): 各个类别下AP的平均值，在本任务中只有一个检测类别，故MAP=AP。

在计数准确率方面，采用均方误差(MSE)为评价指标。
11. x y h w c  c为检测置信度，范围0-1。

