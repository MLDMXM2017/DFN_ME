#! /bin/bash

time=$(date +%Y:%m:%d-%H:%M:%S)
echo 'apex 1 start at' $time
python baseline_flow_extract.py --model examples/demo_mnist-gc.json --save gc --data_num 1 --prefix 0

time=$(date +%Y:%m:%d-%H:%M:%S)
echo 'apex 2 start at' $time
python baseline_flow_extract.py --model examples/demo_mnist-gc.json --save gc --data_num 2 --prefix 0

time=$(date +%Y:%m:%d-%H:%M:%S)
echo 'apex 3 start at' $time
python baseline_flow_extract.py --model examples/demo_mnist-gc.json --save gc --data_num 3 --prefix 0

time=$(date +%Y:%m:%d-%H:%M:%S)
echo 'apex 4 start at' $time
python baseline_flow_extract.py --model examples/demo_mnist-gc.json --save gc --data_num 4 --prefix 0

time=$(date +%Y:%m:%d-%H:%M:%S)
echo 'apex 5 start at' $time
python baseline_flow_extract.py --model examples/demo_mnist-gc.json --save gc --data_num 5 --prefix 0