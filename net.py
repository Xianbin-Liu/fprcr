import torch
import torch.nn as nn

from typing import Dict

class CondNet(nn.Module):
    def __init__(self, model_in=32768, ffn_dim_base=1000, dim1_drop=0.5, 
                 ffn_dim1=100, ffn_dim2=100, 
                 out={'c': 54, 's1': 85, 's2':41, 'r1': 223, 'r2': 95}):
        super(CondNet, self).__init__()
        self.rxn_base = int(ffn_dim_base * dim1_drop)
        inter_out =  self.rxn_base
        self.cat_drop = nn.Sequential(
            nn.Linear(model_in, ffn_dim_base),
            nn.ReLU(),
            nn.Linear(ffn_dim_base, ffn_dim_base),
            nn.ReLU(), 
            nn.Linear(ffn_dim_base, inter_out)
        )
        
        self.cat_next = nn.Sequential(
            nn.Linear(inter_out, ffn_dim2),
            nn.ReLU(),
            nn.Linear(ffn_dim2, ffn_dim2),
            nn.Tanh(),
            nn.Linear(ffn_dim2, out['c'])
        )

        inter_out += ffn_dim2
        self.s1_base, self.s1_out = \
            self.build_cat(out['c'], ffn_dim1, ffn_dim2, out['s1'], inter_out)
        
        inter_out += ffn_dim2
        self.s2_base, self.s2_out = \
            self.build_cat(out['s1'], ffn_dim1, ffn_dim2, out['s2'], inter_out)
        
        inter_out += ffn_dim2
        self.r1_base, self.r1_out = \
            self.build_cat(out['s2'], ffn_dim1, ffn_dim2, out['r1'], inter_out)
        
        inter_out += ffn_dim2
        self.r2_base, self.r2_out = \
            self.build_cat(out['r1'], ffn_dim1, ffn_dim2, out['r2'], inter_out)

    def build_cat(self, model_in, ffn_dim1, ffn_dim2, out_dim, inter_out):
        model_base = nn.Sequential(
            nn.Linear(model_in, ffn_dim1),
            nn.ReLU(),
        )

        model_out = nn.Sequential(
            nn.Linear(inter_out, ffn_dim2),
            nn.ReLU(),
            nn.Linear(ffn_dim2, ffn_dim2),
            nn.Tanh(),
            nn.Linear(ffn_dim2, out_dim)
        )

        return model_base, model_out
    
    def forward(self, Xr, Xp):
        cat_base = torch.cat([Xr, Xp], dim=1)
        cat_base = self.cat_drop(cat_base)
        c = self.cat_next(cat_base)

        s1_base = self.s1_base(c)
        cat_base = torch.cat([cat_base, s1_base], dim=1)
        s1 = self.s1_out(cat_base)

        s2_base = self.s2_base(s1)
        cat_base = torch.cat([cat_base, s2_base], dim=1)
        s2 = self.s2_out(cat_base)

        r1_base = self.r1_base(s2)
        cat_base = torch.cat([cat_base, r1_base], dim=1)
        r1 = self.r1_out(cat_base)

        r2_base = self.r2_base(r1)
        cat_base = torch.cat([cat_base, r2_base], dim=1)
        r2 = self.r2_out(cat_base)

        return {'c': c, 's1': s1, 's2': s2, 'r1': r1, 'r2': r2}

net = CondNet()
