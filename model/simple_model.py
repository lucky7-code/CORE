import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy


class Per_Concept_bias(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return

class Concept_Only(nn.Module):
    def __init__(self, c_num,q_num):
        super().__init__()
        self.num_c = c_num
        self.c_emb = nn.Embedding(c_num, 128)
        self.q_emb = nn.Embedding(q_num, 128)
        self.net = nn.Sequential(
            # nn.Linear(128, 64),
            nn.Linear(128, 32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        q, c, r, t = data["qseqs"].long(), data["cseqs"].long(), data["rseqs"].long(), data["tseqs"].long()
        qshft, cshft, rshft, tshft = data["shft_qseqs"], data["shft_cseqs"], data["shft_rseqs"], data["shft_tseqs"]
        m, sm = data["masks"], data["smasks"]
        # c = self.c_emb(cshft)
        q = self.q_emb(qshft)
        # y = self.net(torch.cat([c,q],-1)).squeeze(-1)
        y = self.net(q).squeeze(-1)
        y = torch.masked_select(y, sm)
        c = torch.masked_select(cshft, sm)
        res = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), res.double())
        return loss, y, res, c + self.num_c * res


if __name__ == '__main__':
    emd = nn.Embedding(11,2,padding_idx=-1)
    data = emd(torch.tensor([1,2,3,4,10,10]))
    print(data)