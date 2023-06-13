import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
import numpy as np

class DKT(Module):
    def __init__(self,num_c, num_q, emb_size, dropout=0.1, emb_type='qid'):
        super().__init__()
        self.model_name = "dkt"
        self.num_c = num_c
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self.c_emb = Embedding(self.num_c+1, self.emb_size, padding_idx=0)
        self.ques_emb = Embedding(self.num_q, self.emb_size)
        self.lstm_layer = LSTM(4*self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_q)

    def multi_skills_embedding(self,c):
        concept_emb_sum = self.c_emb(c + 1).sum(-2)
        # [batch_size, seq_len,1]
        concept_num = torch.where(c < 0, 0, 1).sum(-1).unsqueeze(-1)
        concept_num = torch.where(concept_num == 0, 1, concept_num)
        concept_avg = (concept_emb_sum / concept_num)
        return concept_avg



    def forward(self, data):
        q, c, r, t = data["qseqs"].long(), data["cseqs"].long(), data["rseqs"].long(), data["tseqs"].long()
        qshft, cshft, rshft, tshft = data["shft_qseqs"], data["shft_cseqs"], data["shft_rseqs"], data["shft_tseqs"]
        m, sm = data["masks"], data["smasks"]

        q_emb = self.ques_emb(q)
        c_emb = self.multi_skills_embedding(c)
        x = torch.cat([q_emb, c_emb], -1)
        r = r.unsqueeze(-1)
        resp_t = r.repeat(1,1,2*self.emb_size)
        resp_f = (1-r).repeat(1,1,2*self.emb_size)
        e_emb = torch.cat([x.mul(resp_t),x.mul(resp_f)], -1)
        h, _ = self.lstm_layer(e_emb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)
        y = (y * one_hot(qshft.long(),self.num_q)).sum(-1)
        y = torch.masked_select(y, sm)
        q = torch.masked_select(qshft, sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double())
        np.savetxt('./res/dkt.csv', y.detach().cpu())
        return loss, y, t,  q + self.num_q * t