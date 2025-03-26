import os
os.system("python run_recbole_gnn.py --dataset='steam' --train_batch_size=256 "
                              "--model='MyModel'  --sim='cos' "
                              "--tau=1 --hidden_dropout_prob=0.9 --attn_dropout_prob=0.9") 