#! /bin/bash

is_argu=true
batch_size=16
learning_rate=0.00001
epochs=500

#batchsize
time=$(date +%Y:%m:%d-%H:%M:%S)
echo 'SimpleCNN—batchsize1 start at '$time
python train_DFN.py --batch_size 64 --learning_rate $learning_rate 

#batchsize
time=$(date +%Y:%m:%d-%H:%M:%S)
echo 'SimpleCNN—batchsize2 start at '$time
python train_DFN.py  --batch_size 16 --learning_rate $learning_rate 

#batchsize
time=$(date +%Y:%m:%d-%H:%M:%S)
echo 'SimpleCNN—batchsize3 start at '$time
python train_DFN.py  --batch_size 8 --learning_rate $learning_rate 

#learning rate
time=$(date +%Y:%m:%d-%H:%M:%S)
echo 'SimpleCNN—rate1 start at '$time
python train_DFN.py  --batch_size $batch_size --learning_rate 0.0001

#batchsize
time=$(date +%Y:%m:%d-%H:%M:%S)
echo 'SimpleCNN—rate2 start at '$time
python train_DFN.py  --batch_size $batch_size --learning_rate 0.001

#batchsize
time=$(date +%Y:%m:%d-%H:%M:%S)
echo 'SimpleCNN—rate3 start at '$time
python train_DFN.py  --batch_size $batch_size --learning_rate 0.01

#batchsize
time=$(date +%Y:%m:%d-%H:%M:%S)
echo 'SimpleCNN—rate4 start at '$time
python train_DFN.py  --batch_size $batch_size --learning_rate 0.1

#batchsize
time=$(date +%Y:%m:%d-%H:%M:%S)
echo 'SimpleCNN—rate4 start at '$time
python train_DFN.py  --batch_size $batch_size --learning_rate 0.05
