import os
os.system("python run_recbole_gnn.py --dataset='ml-1m' --train_batch_size=256 "
                              "--model='MyModel' "
                              "--tau=1 --hidden_dropout_prob=0.1 --attn_dropout_prob=0.1 --user_inter_num_interval='[5,inf)' --item_inter_num_interval='[5,inf)'") 
