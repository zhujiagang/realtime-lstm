CUDA_VISIBLE_DEVICES=0 /home/user/anaconda3/bin/python train-ucf24.py --data_root=/home/zhujiagang/realtime/ --save_root=/home/zhujiagang/realtime-lstm/save 
--visdom=True --input_type=rgb --stepvalues=70000,90000 --max_iter=120000
