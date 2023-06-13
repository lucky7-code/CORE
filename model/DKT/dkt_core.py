import torch
import torch.nn as nn
from torch.nn import Module, Embedding, LSTM, Dropout,Parameter
import numpy as np

eps = 1e-12
class DKT_CORE(Module):
    def __init__(self,num_c, num_q, emb_size, dropout=0.1, emb_type='qid'):
        super().__init__()
        self.device = "cuda"
        self.num_c = num_c
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self.c_emb = Embedding(self.num_c+1, self.emb_size, padding_idx=0)
        self.ques_emb = Embedding(self.num_q, self.emb_size)
        self.lstm_layer = LSTM(4*self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.fusion_mode = 'sum'
        self.constant = Parameter(torch.tensor(0.0))
        self.fc = nn.Sequential(
            nn.Linear(3 * self.emb_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.q_nn = nn.Sequential(
            nn.Linear(self.emb_size*2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.s_nn = nn.Linear(self.emb_size, 2)
        self.softmax = nn.Softmax(-1)
        self.loss = nn.CrossEntropyLoss()

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

        qshft_emb = self.ques_emb(qshft)
        cshft_emb = self.multi_skills_embedding(cshft)
        xshft = torch.cat([qshft_emb, cshft_emb], -1)
        out = self.fc(torch.cat([h,xshft],-1))

        q_logit = self.q_nn(xshft.detach())
        s_logit = self.s_nn(h.detach())
        logits = out
        # both q, k and v are the facts
        z_qks = self.fusion(logits, q_logit, s_logit, q_fact=True, k_fact=True, s_fact=True)
        # q is the fact while k and v are the counter_factual
        z_q = self.fusion(logits, q_logit, s_logit, q_fact=True, k_fact=False, s_fact=False)
        # z_sk = self.fusion(logits, q_logit, s_logit, q_fact=False, k_fact=True, s_fact=True)
        # z = self.fusion(logits, q_logit, s_logit, q_fact=False, k_fact=False, s_fact=False)
        logit_Core_DKT = z_qks - z_q
        # TIE
        z_nde = self.fusion(logits.clone().detach(), q_logit.clone().detach(), s_logit.clone().detach(),
                            q_fact=True, k_fact=False, s_fact=False)
        # NDE = z_q - z
        sm_ = sm.unsqueeze(-1)
        z_nde_pred = torch.masked_select(z_nde, sm_).view(-1,2)
        q_pred = torch.masked_select(q_logit,sm_).view(-1,2)
        s_pred = torch.masked_select(s_logit,sm_).view(-1,2)
        z_qks_pred = torch.masked_select(z_qks,sm_).view(-1,2)
        Core_DKT_pred = torch.masked_select(logit_Core_DKT, sm_).view(-1,2)

        # """origin"""
        # logits_pred = torch.masked_select(logits, sm_).view(-1, 2)

        q = torch.masked_select(qshft, sm)
        t = torch.masked_select(rshft, sm).long()
        
        loss_cls = self.loss(z_qks_pred, t) + self.loss(q_pred, t)
        p_te = self.softmax(z_qks_pred).clone().detach()
        loss_kl = - p_te * self.softmax(z_nde_pred).log()
        loss_kl = loss_kl.sum(1).mean()
        loss = loss_cls + loss_kl
        return loss, self.softmax(Core_DKT_pred)[:, 1], t,  q + self.num_q * t


    def multi_skills_embedding(self,c):
        concept_emb_sum = self.c_emb(c + 1).sum(-2)
        # [batch_size, seq_len,1]
        concept_num = torch.where(c < 0, 0, 1).sum(-1).unsqueeze(-1)
        concept_num = torch.where(concept_num == 0, 1, concept_num)
        concept_avg = (concept_emb_sum / concept_num)
        return concept_avg

    def fusion(self, z_k, z_q, z_s, q_fact=False, k_fact=False, s_fact=False):

        global z
        z_k, z_q, z_s = self.transform(z_k, z_q, z_s, q_fact, k_fact, s_fact)

        if self.fusion_mode == 'rubi':
            z = z_k * torch.sigmoid(z_q)

        elif self.fusion_mode == 'hm':
            z = z_k * z_s * z_q
            z = torch.log(z + eps) - torch.log1p(z)

        elif self.fusion_mode == 'sum':
            z = z_k + z_q + z_s
            z = torch.log(torch.sigmoid(z) + eps)
        return z

    def transform(self, z_k, z_q, z_s, q_fact=False, k_fact=False, s_fact=False):

        if not k_fact:
            z_k = self.constant * torch.ones_like(z_k).to(self.device)

        if not q_fact:
            z_q = self.constant * torch.ones_like(z_q).to(self.device)

        if not s_fact:
            z_s = self.constant * torch.ones_like(z_s).to(self.device)

        if self.fusion_mode == 'hm':
            z_k = torch.sigmoid(z_k)
            z_q = torch.sigmoid(z_q)
            z_s = torch.sigmoid(z_s)

        return z_k, z_q, z_s