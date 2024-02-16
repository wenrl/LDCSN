# LDCSN
LDCSN: Lightweight Dual-branch Convolutional Self-attention Network for Masked Face Recognition

# Test pretained model
python test.py 
--cfg
./experiments/CASIA-112x96-LMDB.yaml
--model
LResNet50_LDCSN
--batch_size
64
--gpus
0
--debug
0
