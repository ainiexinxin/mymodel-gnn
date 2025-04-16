from logging import getLogger
import math
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from recbole.data.interaction import Interaction


def gnn_construct_transform(config, phase):
    if config['gnn_transform'] is None:
        raise ValueError('config["gnn_transform"] is None but trying to construct transform.')
    str2transform = {
        'sess_graph': SessionGraph,
    }
    return str2transform[config['gnn_transform']](config, phase)


class SessionGraph:
    def __init__(self, config, phase):
        self.config = config
        self.phase = phase
        self.n_items = config['n_items']
        self.logger = getLogger()
        self.logger.info('SessionGraph Transform in DataLoader.')

    def __call__(self, dataset, interaction):
        graph_objs = dataset.graph_objs
        index = interaction['graph_idx']
        graph_batch = {
            k: [graph_objs[k][_.item()] for _ in index]
            for k in graph_objs
        }
        graph_batch['batch'] = []

        tot_node_num = torch.ones([1], dtype=torch.long)
        for i in range(index.shape[0]):
            for k in graph_batch:
                if 'edge_index' in k:
                    graph_batch[k][i] = graph_batch[k][i] + tot_node_num
            if 'alias_inputs' in graph_batch:
                graph_batch['alias_inputs'][i] = graph_batch['alias_inputs'][i] + tot_node_num
            graph_batch['batch'].append(torch.full_like(graph_batch['x'][i], i))
            tot_node_num += graph_batch['x'][i].shape[0]

        if hasattr(dataset, 'node_attr'):
            node_attr = ['batch'] + dataset.node_attr
        else:
            node_attr = ['x', 'batch']
        for k in node_attr:
            graph_batch[k] = [torch.zeros([1], dtype=graph_batch[k][-1].dtype)] + graph_batch[k]

        for k in graph_batch:
            if k == 'alias_inputs':
                graph_batch[k] = pad_sequence(graph_batch[k], batch_first=True)
            else:
                graph_batch[k] = torch.cat(graph_batch[k], dim=-1)
        
        if self.phase == 'train':
            self.augmentation(interaction)
        elif self.config['test_noise_ratio'] > 0:
            self.test_noise(interaction, self.config['test_noise_ratio'])
        interaction.update(Interaction(graph_batch))
        return interaction
    
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
            
        cur_data.update(Interaction({'aug1': torch.stack(aug_seq1), 'aug_len1': torch.stack(aug_len1),
                                     'aug2': torch.stack(aug_seq2), 'aug_len2': torch.stack(aug_len2)}))
        
    def test_noise(self, cur_data, noise_r):
        def item_mask(seq, length, gamma=noise_r):
            num_mask = math.floor(length * gamma)
            mask_index = random.sample(range(length), k=num_mask)
            masked_item_seq = seq[:]
            for index in mask_index:
                masked_item_seq[index] = random.randint(1,self.n_items)  # token 0 has been used for semantic masking
            return masked_item_seq, length

        seqs = cur_data['item_id_list']
        lengths = cur_data['item_length']

        noise_seq = []
        noise_len = []

        for seq, length in zip(seqs, lengths):
            aug_seq, aug_len = item_mask(seq, length)

            noise_seq.append(aug_seq)
            noise_len.append(aug_len)

        cur_data.update(Interaction({'item_id_list': torch.stack(noise_seq), 'item_length': torch.stack(noise_len)}))
