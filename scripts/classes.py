################################################################################
# MultipleHiddenLayersModel - Multi-Layer Perceptron Neural Networks (MLPs) 
################################################################################
import torch
import torch.nn as nn
from functions import *

class MultipleHiddenLayersModel(nn.Module):
    def __init__(self, layer_sizes, dropout=0.1):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.norms = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(len(self.layer_sizes) - 1):
            if i < len(self.layer_sizes) - 2:
                self.norms.append(nn.SyncBatchNorm(self.layer_sizes[i]))
            self.layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))
            nn.init.xavier_normal_(self.layers[i].weight)
        self.function_for_hidden = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, u, t):
        out = torch.cat((u, t), dim=1)
        for i in range(len(self.layer_sizes) - 2):
            out = self.norms[i](out)
            out = self.layers[i](out)
            out = self.function_for_hidden(out)
            out = self.dropout(out)
        out = self.layers[-1](out)
        out = torch.sigmoid(out)
        return out

from torch.utils.data import Dataset
class SegmentDataset(Dataset):
    def __init__(self, utterance_scp_file_full_path, transcript_scp_file_full_path, label_file_full_path, rank):
        self.utterance_scp_file_full_path = utterance_scp_file_full_path
        self.transcript_scp_file_full_path = transcript_scp_file_full_path
        self.label_file_full_path = label_file_full_path
        self.rank = rank

        self.utterance_feature_dict = read_feature_file(self.utterance_scp_file_full_path, self.rank)
        self.transcript_feature_dict = read_feature_file(self.transcript_scp_file_full_path, self.rank)

        utterance_stm_id_set = set(list(self.utterance_feature_dict.keys()))
        transcript_stm_id_set = set(list(self.transcript_feature_dict.keys()))
        stm_ids_common = utterance_stm_id_set & transcript_stm_id_set
        self.label_dict = read_label_file(self.label_file_full_path, stm_ids_common, self.rank)

        self.zero_wer_count = self._get_zero_wer_count()
        self.stm_id_list = self._remove_zero_wer()
        if is_main_process(self.rank):
            logger.debug(f'Dataset: {len(self.stm_id_list)}')

    def __getitem__(self, index):
        stm_id = self.stm_id_list[index]
        return stm_id, torch.tensor(self.utterance_feature_dict[stm_id]), torch.tensor(self.transcript_feature_dict[stm_id]), torch.tensor(self.label_dict[stm_id]['tkn_er']), torch.tensor(self.label_dict[stm_id]['sub_er']), torch.tensor(self.label_dict[stm_id]['del_er']), torch.tensor(self.label_dict[stm_id]['ins_er'])

    def __len__(self):
        return len(self.stm_id_list)

    def get_stm_id_list(self):
        return self.stm_id_list

    def _get_zero_wer_count(self):
        import numpy as np
        tkn_er_list = [self.label_dict[key]['tkn_er'] for key in self.label_dict]
        counts, bins = np.histogram(tkn_er_list, bins=100, range=[0, 1])
        zipped = zip(counts, bins)
        zipped = list(zipped)
        res = sorted(zipped, key = lambda x: x[0], reverse=True)
        if is_main_process(self.rank):
            logger.info(f'bins: {len(res)}, res[:3]: {res[:3]}, num of WER 0: {res[1][0] + res[2][0]}, total: {len(tkn_er_list)}, remove: {res[0][0] - (res[1][0] + res[2][0])}')
        return res[1][0] + res[2][0]

    def _remove_zero_wer(self):
        stm_id_list = list(self.label_dict.keys())
        for stm_id in stm_id_list:
            if self.label_dict[stm_id]['tkn_er'] == 0:
                if self.zero_wer_count > 0:
                    self.zero_wer_count -= 1
                elif self.zero_wer_count == 0:
                    del self.label_dict[stm_id]
                    continue
        return list(self.label_dict.keys())
