import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Parameter, Embedding, Linear, Dropout
from torch.nn.functional import binary_cross_entropy
from torch.nn.init import kaiming_normal_
eps = 1e-12

class DKVMN_CORE(Module):
    def __init__(self,num_c, num_q, dim_s, size_m, dropout=0.2, emb_type='qid'):
        super().__init__()
        self.model_name = "dkvmn"
        self.num_c = num_c
        self.num_q = num_q
        self.dim_s = dim_s
        self.size_m = size_m
        self.emb_type = emb_type

        self.k_emb_layer = Linear(2*self.dim_s, self.dim_s)
        self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
        self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.v_emb_layer = Linear(4*self.dim_s, self.dim_s)

        self.f_layer = Linear(self.dim_s * 2, self.dim_s)
        self.dropout_layer = Dropout(dropout)
        self.p_layer = Linear(self.dim_s, 2)

        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)
        self.c_emb = Embedding(self.num_c + 1, self.dim_s, padding_idx=0)
        self.q_emb = Embedding(self.num_q, self.dim_s)
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)
        self.constant = nn.Parameter(torch.tensor(0.0))
        self.fusion_mode = 'sum'
        self.device = "cuda"
        self.q_predict_linear = nn.Linear(self.dim_s, 2, bias=True)
        self.s_predict_linear = nn.Linear(self.dim_s, 2, bias=True)
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
        cq = torch.cat((q[:, 0:1], qshft), dim=1).long()
        cc = torch.cat((c[:, 0:1], cshft), dim=1).long()
        cr = torch.cat((r[:, 0:1], rshft), dim=1).long()

        batch_size = q.shape[0]
        q_emb = self.q_emb(cq)
        c_emb = self.multi_skills_embedding(cc)
        x = torch.cat([q_emb, c_emb], -1)

        cr = cr.unsqueeze(-1)
        resp_t = cr.repeat(1,1,2*self.dim_s)
        resp_f = (1-cr).repeat(1,1,2*self.dim_s)
        e_emb = torch.cat([x.mul(resp_t),x.mul(resp_f)], -1)

        k = self.k_emb_layer(x)
        v = self.v_emb_layer(e_emb)

        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)

        Mv = [Mvt]

        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        # Write Process
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))

        for et, at, wt in zip(
                e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                  (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)

        Mv = torch.stack(Mv, dim=1)


        # Read Process
        read_value=(w.unsqueeze(-1) * Mv[:, :-1]).sum(-2)
        f = torch.tanh(self.f_layer(torch.cat([read_value, k],dim=-1)))

        p = self.p_layer(self.dropout_layer(f))
        logits = p
        q_logit = self.q_predict_linear(k.detach())
        s_logit = self.s_predict_linear(read_value.detach())



        z_qks = self.fusion(logits, q_logit, s_logit, q_fact=True, k_fact=True, s_fact=True)
        # q is the fact while k and v are the counter_factual
        z_q = self.fusion(logits, q_logit, s_logit, q_fact=True, k_fact=False, s_fact=False)
        # z_sk = self.fusion(logits, q_logit, s_logit, q_fact=False, k_fact=True, s_fact=True)
        # z = self.fusion(logits, q_logit, s_logit, q_fact=False, k_fact=False, s_fact=False)

        logit_Core_DKVMN = z_qks - z_q  # TIE
        z_nde = self.fusion(logits.clone().detach(), q_logit.clone().detach(), s_logit.clone().detach(),
                            q_fact=True, k_fact=False, s_fact=False)

        sm_ = sm.unsqueeze(-1)
        z_nde_pred = torch.masked_select(z_nde[:,1:,:], sm_).view(-1, 2)
        q_pred = torch.masked_select(q_logit[:,1:,:], sm_).view(-1, 2)
        s_pred = torch.masked_select(s_logit[:,1:,:], sm_).view(-1, 2)
        z_qks_pred = torch.masked_select(z_qks[:,1:,:], sm_).view(-1, 2)
        Core_DKT_pred = torch.masked_select(logit_Core_DKVMN[:,1:,:], sm_).view(-1, 2)


        q = torch.masked_select(qshft, sm)
        t = torch.masked_select(rshft, sm).long()
        loss_cls = self.loss(q_pred, t) + self.loss(z_qks_pred, t)

        p_te = self.softmax(z_qks_pred).clone().detach()
        loss_kl = - p_te * self.softmax(z_nde_pred).log()
        loss_kl = loss_kl.sum(1).mean()
        loss = loss_cls + loss_kl

        return loss, self.softmax(Core_DKT_pred)[:, 1], t,  q+self.num_q * t



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