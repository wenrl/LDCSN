# LDCSN
LDCSN: Lightweight Dual-branch Convolutional Self-attention Network for Masked Face Recognition

#Train model
python train.py
--cfg
./experiments/CASIA-112x112-LDCSN.yaml
--model
"LResNet50_LDCSN"
--batch_size
32
--gpus
"0"
--debug
0

# Test pretained model
python test.py 
--cfg
./experiments/CASIA-112x112-LDCSN.yaml
--model
"LResNet50_LDCSN"
--batch_size
64
--gpus
"0"
--debug
0

# Pretrained model
[预训练模型链接：](https://pan.baidu.com/s/1WT1IANT8nf5mPacvahjpVA)
提取码：1q4g
