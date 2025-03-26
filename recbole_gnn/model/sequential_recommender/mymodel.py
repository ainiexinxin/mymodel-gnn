# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import math
import random
from torch.nn import Parameter
import numpy as np
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
import torch.fft as fft
import torch.nn.functional as F
from recbole_gnn.model.layers import SRGNNCell
from torch_geometric.utils import dropout_edge, dropout_node, dropout_path

class MyModel(SequentialRecommender):
    def __init__(self, config, dataset):
        super(MyModel, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.batch_size = config['train_batch_size']
        self.lmd = config['lmd']
        self.lmd_tf = config['lmd_tf']
        self.tau = config['tau']
        self.sim = config['sim']

        self.tau_plus = config['tau_plus']
        self.beta = config['beta']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        self.step = config['step']
        self.device = config['device']
        self.g_weight = config['g_weight']
        self.noise_base = config['noise_base']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.fft_layer = BandedFourierLayer(self.hidden_size, self.hidden_size, 0, 1, length=self.max_seq_length)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.nce_fct = nn.CrossEntropyLoss()

        self.gnn = SRGNNCell(self.hidden_size)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_3 = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_out = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.dropout_g = nn.Dropout(0.1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def augmentation(self, cur_data):
        def item_crop(seq, length, eta=0.6):
            seq_cpu = seq.clone().cpu()
            num_left = math.floor(length * eta)
            crop_begin = random.randint(0, length - num_left)
            croped_item_seq = np.zeros(seq_cpu.shape[0])
            if crop_begin + num_left < seq_cpu.shape[0]:
                croped_item_seq[:num_left] = seq_cpu[crop_begin:crop_begin + num_left]
            else:
                croped_item_seq[:num_left] = seq_cpu[crop_begin:]
            return torch.tensor(croped_item_seq, dtype=torch.long, device=seq.device), torch.tensor(num_left, dtype=torch.long, device=seq.device)
        
        def item_mask(seq, length, gamma=0.3):
            num_mask = math.floor(length * gamma)
            mask_index = random.sample(range(length), k=num_mask)
            masked_item_seq = seq[:]
            masked_item_seq[mask_index] = self.n_items  # token 0 has been used for semantic masking
            return masked_item_seq, length
        
        def item_reorder(seq, length, beta=0.6):
            num_reorder = math.floor(length * beta)
            reorder_begin = random.randint(0, length - num_reorder)
            reordered_item_seq = seq[:]
            shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
            random.shuffle(shuffle_index)
            reordered_item_seq[reorder_begin:reorder_begin + num_reorder] = reordered_item_seq[shuffle_index]
            return reordered_item_seq, length
        
        seqs = cur_data['item_id_list'].clone()
        lengths = cur_data['item_length'].clone()

        aug_seq1 = []
        aug_len1 = []
        aug_seq2 = []
        aug_len2 = []
        for seq, length in zip(seqs, lengths):
            if length > 5:
                switch = random.sample(range(3), k=2)
            elif 1 < length < 5:
                switch = [2, 2]
            else:
                switch = [3, 3]
                aug_seq = seq
                aug_len = length
            if switch[0] == 0:
                aug_seq, aug_len = item_crop(seq, length)
            elif switch[0] == 1:
                aug_seq, aug_len = item_mask(seq, length)
            elif switch[0] == 2:
                aug_seq, aug_len = item_reorder(seq, length)
    
            aug_seq1.append(aug_seq)
            aug_len1.append(aug_len)
    
            if switch[1] == 0:
                aug_seq, aug_len = item_crop(seq, length)
            elif switch[1] == 1:
                aug_seq, aug_len = item_mask(seq, length)
            elif switch[1] == 2:
                aug_seq, aug_len = item_reorder(seq, length)
    
            aug_seq2.append(aug_seq)
            aug_len2.append(aug_len)
            
        return torch.stack(aug_seq1), torch.stack(aug_len1), torch.stack(aug_seq2), torch.stack(aug_len2)
    
    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, interaction, item_seq, item_seq_len, disturb=False):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)    

        if disturb:
            noise = self.gaussian_noise(item_emb, self.noise_base)
            mask1 = item_seq.gt(0).unsqueeze(dim=2)
            noise1 = noise * mask1
            item_emb = item_emb + noise1
            
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        if disturb:
            input_emb = self.fft_layer(input_emb)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output_t = trm_output[-1]
        output_t = self.gather_indexes(output_t, item_seq_len - 1)

        return output_t
    
    def forward_gcn(self, interaction, item_seq, item_seq_len, disturb=False):
        x = interaction['x']
        edge_index = interaction['edge_index']
        alias_inputs = interaction['alias_inputs']

        mask = alias_inputs.gt(0)
        hidden = self.item_embedding(x)

        old_hidden = hidden.clone()
        if disturb:
            noise = self.gaussian_noise(hidden, self.noise_base)
            mask1 = x.gt(0).unsqueeze(dim=1)
            noise1 = noise * mask1
            hidden = hidden + noise1
            rand = random.sample(range(3), 1)
            if rand == 0:
                edge_droped, _ = dropout_edge(edge_index)
            elif rand == 1:
                edge_droped, _, _ = dropout_node(edge_index)
            else:
                edge_droped, _ = dropout_path(edge_index)
            hidden = self.gnn(self.dropout_g(hidden), edge_droped) + old_hidden
        else:
            hidden = self.gnn(self.dropout_g(hidden), edge_index) + old_hidden
        seq_hidden = hidden[alias_inputs]

        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
        l1 = self.linear_1(ht).view(ht.size(0), 1, ht.size(1))
        l2 = self.linear_2(seq_hidden)

        alpha = self.linear_3(torch.sigmoid(l1 + l2))
        a = torch.sum(alpha * seq_hidden * mask.view(mask.size(0), -1, 1).float(), 1)
        seq_output = self.linear_out(torch.cat([a, ht], dim=1))
        return seq_output
    
    def my_fft(self, seq):
        f = torch.fft.rfft(seq, dim=1)
        amp = torch.absolute(f)
        phase = torch.angle(f)
        return amp, phase
    
    def calculate_loss(self, interaction):
        # cal loss
        loss = torch.tensor(0.0).to(self.device)
        loss += 0.5 * self.rec_loss(interaction, self.forward)
        loss += 0.5 * self.rec_loss(interaction, self.forward_gcn)
        loss += self.cl_loss(interaction)
        return loss
    
    def cl_loss(self, interaction):
        loss = torch.tensor(0.0).to(self.device)
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        aug_seq1, aug_len1, aug_seq2, aug_len2 = self.augmentation(interaction)
        tf_seq_output_f_1 = self.forward(interaction, aug_seq1, aug_len1, True)
        tf_seq_output_f_2 = self.forward(interaction, aug_seq2, aug_len2, True)
        gnn_seq_output_1 = self.forward_gcn(interaction, item_seq, item_seq_len, True)
        gnn_seq_output_2 = self.forward_gcn(interaction, item_seq, item_seq_len, True)

        # nce_logits, nce_labels = self.info_nce(tf_seq_output_1, tf_seq_output_2, temp=self.tau, batch_size=tf_seq_output_1.shape[0], sim=self.sim)
        # loss += 0.01 * self.nce_fct(nce_logits, nce_labels)

        loss += 0.01 * self.infonce(tf_seq_output_f_1, tf_seq_output_f_2, temp=self.tau, batch_size=tf_seq_output_f_1.shape[0])

        loss += 0.01 * self.infonce(gnn_seq_output_1, gnn_seq_output_2, temp=self.tau, batch_size=gnn_seq_output_1.shape[0])

        return loss
    
    def rec_loss(self, interaction, forward):
        # item_seq
        item_seq = interaction[self.ITEM_SEQ]  # N(B) * L(sequence length), long truncation, not enough to fill 0
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        # seq_output It is the result of passing the input sequence forward through the neural network. It is a tensor of shape N * D, where N is the batch size and D is the dimension of the output
        seq_output = forward(interaction, item_seq, item_seq_len)
        # pos_items Represents a positive item associated with a given sequence.
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            # Calculate Bayesian Personalized Ranking (BPR) losses, using positive and negative items and embeddings to calculate scores, and then calculate losses
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type == 'CE'
            # 'CE'(cross entropy), then calculate the loss by embedding the item, using the softmax function
            # item_emb get all
            test_item_emb = self.item_embedding.weight
            # seq_output and all item_emb do similarity calculation # tensor.transpose exchange matrix of two dimensions, transpose(dim0, dim1)
            # torch.matmul is multiplication of tensor,
            # Input can be high dimensional. 2D is normal matrix multiplication like tensor.mm.
            # When there are multiple dimensions, the extra one dimension is brought out as batch, and the other parts do matrix multiplication
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
            loss = self.loss_fct(logits, pos_items)

        if torch.isnan(loss) or torch.isinf(loss):
            print("rec_loss:", loss)
            loss = 1e-8
        return loss
    
    def infonce(self, z_i, z_j, temp, batch_size):  # B * D    B * D  (batch_size * dim)
        # Positive sample: It is the inner product of itself and itself, or the product of the NTH row in z_i and the NTH row in z_j.
        # Negative sample: is the inner product of itself and itself, or the product of the zeroth row in z_i and the zeroth row in z_j.
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        if self.sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif self.sim == 'dot':
            sim = torch.mm(z, z.T) / temp

        # Take the diagonal entries of the matrix
        sim_i_j = torch.diag(sim, batch_size)  # B*1 # The main diagonal must be moved B rows up to be a positive sample
        sim_j_i = torch.diag(sim, -batch_size)  # B*1 # The main diagonal must be moved B rows up to be a positive sample

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # build negative samples
        # mask
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_correlated_samples(batch_size=self.batch_size)
        negative_samples = sim[mask].reshape(N, -1)  

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # N * C
        logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
        loss = self.nce_fct(logits+1e-8, labels)  # BPRLoss or CrossEntropyLoss(Cross entropy loss)
        if torch.isnan(loss) or torch.isinf(loss):
            print("cl_loss_1:", loss)
            loss = 1e-8
        return loss

    
    def gaussian_noise(self, source, noise_base=0.1, dtype=torch.float32):
        x = noise_base + torch.zeros_like(source, dtype=dtype, device=source.device)
        noise = torch.normal(mean=torch.tensor([0.0]).to(source.device), std=x).to(source.device)
        return noise

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    
    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output_t = self.forward(interaction, item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output_t, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output_t = self.forward(interaction,item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
        scores = torch.matmul(seq_output_t, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores


class BandedFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, band, num_bands, length=201):
        super().__init__()

        self.length = length
        self.total_freqs = (self.length // 2) + 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.band = band  # zero indexed
        self.num_bands = num_bands

        self.num_freqs = self.total_freqs // self.num_bands + (
            self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0)

        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs

        # case: from other frequencies
        self.weight = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat))
        self.reset_parameters()

    def forward(self, input):
        # input - b t d
        b, t, _ = input.shape
        input_fft = fft.rfft(input, dim=1)
        output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, device=input.device, dtype=torch.cfloat)
        output_fft[:, self.start:self.end] = self._forward(input_fft)
        return fft.irfft(output_fft, n=input.size(1), dim=1)

    def _forward(self, input):
        output = torch.einsum('bti,tio->bto', input[:, self.start:self.end], self.weight)
        return output + self.bias

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
