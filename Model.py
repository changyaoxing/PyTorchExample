from torch import nn
class Perception(nn.Module):
    def __init__(self, in_dim, hid_dim1,hid_dim2,hid_dim3, out_dim):
        super(Perception,self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim,hid_dim1),
            nn.Sigmoid(),
            nn.Linear(hid_dim1,hid_dim2),
            nn.Sigmoid(),
            nn.Linear(hid_dim2, hid_dim3),
            nn.Sigmoid(),
            nn.Linear(hid_dim3, out_dim),
            nn.Sigmoid(),
        )
    def forward(self,x):
        y=self.layer(x)
        return y