import torch

from torch.nn import Module, Embedding, Linear, MultiheadAttention, LayerNorm, Dropout
from pykt.models.utils import transformer_FFN, pos_encode, ut_mask, get_clones
from torch.nn.functional import binary_cross_entropy


class SAKT(Module):
    def __init__(self, num_c, num_q, seq_len, emb_size, num_attn_heads, dropout, num_en=2, emb_type="qid"):
        super().__init__()
        self.model_name = "sakt"
        self.emb_type = emb_type

        self.num_c = num_c
        self.num_q = num_q
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_en = num_en

        # num_c, seq_len, emb_size, num_attn_heads, dropout, emb_path="")
        self.q_emb = Embedding(num_q, emb_size)
        self.c_emb = Embedding(num_c+1, emb_size, padding_idx=0)
        # self.P = Parameter(torch.Tensor(self.seq_len, self.emb_size))
        self.position_emb = Embedding(seq_len, 4*emb_size)

        self.blocks = get_clones(Blocks(emb_size, num_attn_heads, dropout), self.num_en)

        self.dropout_layer = Dropout(dropout)
        self.pred = Linear(self.emb_size, 1)
        self.x_emb = Linear(2*emb_size, emb_size)
        self.e_emb = Linear(4*emb_size, emb_size)

    def multi_skills_embedding(self,c):
        concept_emb_sum = self.c_emb(c + 1).sum(-2)
        # [batch_size, seq_len,1]
        concept_num = torch.where(c < 0, 0, 1).sum(-1).unsqueeze(-1)
        concept_num = torch.where(concept_num == 0, 1, concept_num)
        concept_avg = (concept_emb_sum / concept_num)
        return concept_avg

    def base_emb(self, q, c, r, qry, cry):
        q_emb = self.q_emb(q)
        c_emb = self.multi_skills_embedding(c)
        x = torch.cat([q_emb, c_emb], -1)
        r = r.unsqueeze(-1)
        resp_t = r.repeat(1, 1, 2 * self.emb_size)
        resp_f = (1 - r).repeat(1, 1, 2 * self.emb_size)
        e_emb = torch.cat([x.mul(resp_t), x.mul(resp_f)], -1)


        qshftemb = torch.cat([self.q_emb(qry), self.multi_skills_embedding(cry)], -1)

        posemb = self.position_emb(pos_encode(e_emb.shape[1]))
        e_emb = e_emb + posemb
        return self.x_emb(qshftemb), self.e_emb(e_emb)

    def forward(self, data):
        q, c, r, t = data["qseqs"].long(), data["cseqs"].long(), data["rseqs"].long(), data["tseqs"].long()
        qshft, cshft, rshft, tshft = data["shft_qseqs"], data["shft_cseqs"], data["shft_rseqs"], data["shft_tseqs"]
        m, sm = data["masks"], data["smasks"]


        qshftemb, xemb = self.base_emb(q, c, r, qshft, cshft)

        # print(f"qemb: {qemb.shape}, xemb: {xemb.shape}, qshftemb: {qshftemb.shape}")
        for i in range(self.num_en):
            xemb = self.blocks[i](qshftemb, xemb, xemb)
        p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)

        y = torch.masked_select(p, sm)
        q = torch.masked_select(qshft, sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double())
        return loss, y, t, q + self.num_q * t



class Blocks(Module):
    def __init__(self, emb_size, num_attn_heads, dropout) -> None:
        super().__init__()

        self.attn = MultiheadAttention(emb_size, num_attn_heads, dropout=dropout)
        self.attn_dropout = Dropout(dropout)
        self.attn_layer_norm = LayerNorm(emb_size)

        self.FFN = transformer_FFN(emb_size, dropout)
        self.FFN_dropout = Dropout(dropout)
        self.FFN_layer_norm = LayerNorm(emb_size)

    def forward(self, q=None, k=None, v=None):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        # attn -> drop -> skip -> norm
        # transformer: attn -> drop -> skip -> norm transformer default
        causal_mask = ut_mask(seq_len=k.shape[0])
        attn_emb, _ = self.attn(q, k, v, attn_mask=causal_mask)

        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)

        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb