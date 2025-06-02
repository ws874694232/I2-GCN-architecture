import torch
import torch.nn as nn
from layers import GraphConvolution


#两层图卷积结构
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.R = nn.ReLU()
        self.B = nn.BatchNorm2d(1)

        self.nfeat1_2 = 1
        self.nhid1_2 = 4

        self.gc1_1 = GraphConvolution(self.nfeat1_2, self.nhid1_2)
        self.gc1_2 = GraphConvolution(self.nhid1_2, self.nfeat1_2)

        self.gc2_1 = GraphConvolution(self.nfeat1_2, self.nhid1_2)
        self.gc2_2 = GraphConvolution(self.nhid1_2, self.nfeat1_2)



    #     #     #     #     #     #     #     #     #     #     #     #
        self.gc2_1_1 = GraphConvolution(2, 4)
        self.gc2_2_1 = GraphConvolution(4, 1)


    def forward(self, x, adj):



        F1 = x[0]
        F2 = x[1]

        A1 = adj[0]
        A2 = adj[1]
        A1_2 = adj[2]



        F1_1 = self.R(self.gc1_1(F1, A1)).permute(1, 0)
        F1_1 = self.B(F1_1.unsqueeze(0).unsqueeze(0))
        F1_1 = F1_1.squeeze(0).squeeze(0)
        F1_1 = F1_1.permute(1, 0)
        F1_2 = self.R(self.gc1_2(F1_1,A1)).permute(1, 0)
        F1_2 = self.B(F1_2.unsqueeze(0).unsqueeze(0))
        F1_2 = F1_2.squeeze(0).squeeze(0)
        F1_2 = self.R(F1_2.permute(1, 0))

        F2_1 = self.R(self.gc2_1(F2, A2)).permute(1, 0)
        F2_1 = self.B(F2_1.unsqueeze(0).unsqueeze(0))
        F2_1 = F2_1.squeeze(0).squeeze(0)
        F2_1 = F2_1.permute(1, 0)
        F2_2 = self.R(self.gc2_2(F2_1, A2)).permute(1, 0)
        F2_2 = self.B(F2_2.unsqueeze(0).unsqueeze(0))
        F2_2 = F2_2.squeeze(0).squeeze(0)
        F2_2 = self.R(F2_2.permute(1, 0))

        #     #     #     #     #     #     #     #     #     #     #     #


        #     #     #     #     #     #     #     #     #     #     #     #
        F = torch.cat([F1_2, F2_2], dim=1)
        F = self.R(self.gc2_1_1(F, A1_2)).permute(1, 0)
        F = self.B(F.unsqueeze(0).unsqueeze(0))
        F = F.squeeze(0).squeeze(0)
        F = F.permute(1, 0)
        out = self.R(self.gc2_2_1(F, A1_2)).permute(1, 0)
        out = self.B(out.unsqueeze(0).unsqueeze(0))
        out = out.squeeze(0).squeeze(0)
        out = self.R(out.permute(1, 0))

        # x = self.gc2(x, adj).permute(1, 0)
        # x = self.conv2(x.unsqueeze(0).unsqueeze(0))
        # x = x.squeeze(0).squeeze(0).permute(1, 0)
        #
        # x = self.gc3(x, adj).permute(1, 0)
        # x = self.conv3(x.unsqueeze(0).unsqueeze(0))
        # x = x.squeeze(0).squeeze(0).permute(1, 0)
        #
        # x = self.gc4(x, adj).permute(1, 0)
        # x = self.conv4(x.unsqueeze(0).unsqueeze(0))
        # x = x.squeeze(0).squeeze(0).permute(1, 0)

        # x = self.r(self.gc1(x, adj))
        # x = self.r(self.gc2(x, adj))
        # x = self.r(self.gc3(x, adj))
        #
        # # x = self.gc3(x, adj)
        # # x = self.gc4(x, adj)
        # # x = self.gc5(x, adj)
        return out





if __name__ == '__main__':
    # x = torch.rand([85, 1944, 3])
    # adj = torch.rand([1944, 1944])
    gcn = GCN()
    # b = gcn(x[0], adj)

    Adj = torch.rand([3, 324, 324])
    F = torch.rand([2, 324, 1])
    out = gcn(F, Adj)
    print(out.shape)



















