import os
os.system("python run_recbole_gnn.py --dataset='sports' --train_batch_size=256 "
                              "--lmd=0.01 --lmd_sem=0.1 --model='MyModel'  --sim='cos' "
                              "--tau=1 --hidden_dropout_prob=0.3 --attn_dropout_prob=0.3 --lmd_tf=0.5") 