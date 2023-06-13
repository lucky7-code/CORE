import os

import numpy as np
import torch

from torch.nn import Module, Parameter, Embedding, Linear, Dropout
from torch.nn.functional import binary_cross_entropy
from torch.nn.init import kaiming_normal_


class DKVMN(Module):
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
        self.p_layer = Linear(self.dim_s, 1)

        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)
        self.c_emb = Embedding(self.num_c + 1, self.dim_s, padding_idx=0)
        self.q_emb = Embedding(self.num_q, self.dim_s)
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
        f = torch.tanh(
            self.f_layer(
                torch.cat(
                    [
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim=-1
                )
            )
        )
        p = self.p_layer(self.dropout_layer(f))
        p = torch.sigmoid(p)
        p = p.squeeze(-1)

        y = torch.masked_select(p[:,1:], sm)
        q = torch.masked_select(qshft, sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double())

        return loss, y, t,  q+self.num_q * t