# CRNN_ATTN_CTC for STR（场景文字识别）
不同于OCR，STR因为其文字展现形式的多样性，其难度远大于扫描文档图像中的文字识别。包括如下：
* 字符是不同数字、语种文字的混合；
* 字符有不同的大小、字体、颜色、亮度、对比度等；
* 文本行有横向、竖向、弯曲、旋转、扭曲等式样；
* 文字区域产生变形(透视、仿射变换)、残缺、模糊等现象；
* 文字周围背景噪声千变万化；  

**所以，该项目主要针对场景文字序列，在CTPN检测得到的文字区域基础上，进行文字序列的学习与识别。**  
<img src="https://github.com/RoyceMao/CRNN_ATTN_CTC/blob/master/out/digits_1.jpg" width="150" height="50" /> <img src="https://github.com/RoyceMao/CRNN_ATTN_CTC/blob/master/out/digits_2.jpg" width="150" height="50" />
## Requirements
* Torch >= 1.0.0   
* Torchvision >= 0.2.0   
* Torchnet >= 0.0.4
* Augmentor 
* OpenCV
* NVIDIA CUDA cudnn  
## KeyPoints
| 1、数据增广 | 2、CRNN结构 | 3、输出序列解码 |
| ------ | ------ | ------ |
| 比例缩放、随机扭曲、随机透视、随机遮挡、色彩抖动等 | ResNet18+BiLSTM+Attention | 分定长或不定长 |
## Getting Started
### Inference
[config.py](/config.py)中指定需要预测的**TEST_PATH**，运行[prediction.py](/prediction.py)脚本：
```
python prediction.py --[params]
```
### Train
#### 1. Data Prepare
首先，[config.py](/config.py)中指定未增广的训练集**TRAIN_PATH**，运行[data_aug.py](/data_aug.py)，用Augmentor包方法实现KeyPoints中候选的增广逻辑，移动生成的output文件夹下，所有训练样本至**AUGED_TRAIN_PATH**：
```
python data_aug.py
```
#### 2. Training
修改[config.py](/config.py)中训练相关的部分参数，执行训练过程：
```
python train.py --[params]
```
#### 3. Logs
以下是场景文字串训练日志样例（非公开数据）。在captcha验证码库生成的1000例数据集上测试，随机2:1切割，测试集的准确率Acc能达到99.60%以上：
```
...
Epoch: 50 Loss: 0.0026 Acc: 0.9989 : 100%|██████████| 28/28 [00:02<00:00, 12.50it/s]
Test : 50 Loss: 0.0345 Acc: 0.9444 Acc_relax: 0.9921 : 100%|██████████| 72/72 [00:01<00:00, 65.47it/s]
```
#### 4. E.G.
<img src="https://github.com/RoyceMao/CRNN_ATTN_CTC/blob/master/out/pred.jpg" width="600" height="100" />    

## Reference
https://xiaodu.io/ctc-explained/   
本次工程代码，CRNN输出序列到最终label序列的解码逻辑，只针对字符定长的场景文本。非定长的代码部分，**会尽快更新...**
