import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import dgl
from dgl.nn import GATConv
from dgl.nn.pytorch.glob import MaxPooling,AvgPooling

class ConvsLayer(torch.nn.Module):

    def __init__(self,emb_dim):
        super(ConvsLayer,self).__init__() 
        self.embedding_size = emb_dim
        self.conv1 = nn.Conv1d(in_channels=self.embedding_size,out_channels = 128, kernel_size = 3)
        self.mx1 = nn.MaxPool1d(3, stride=3)
        self.conv2 = nn.Conv1d(in_channels=128,out_channels = 128, kernel_size = 3)
        self.mx2 = nn.MaxPool1d(3, stride=3)
        self.conv3 = nn.Conv1d(in_channels=128,out_channels = 128, kernel_size = 3)
        self.mx3 = nn.MaxPool1d(130, stride=1)
    
    def forward(self,x):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        features = self.conv1(x)       
        features = self.mx1(features)
        features = self.mx2(self.conv2(features))
        features = self.conv3(features)
        features = self.mx3(features)
        features = features.squeeze(2)
        return features


class GATPPI(torch.nn.Module):

    def __init__(self,args):
        super(GATPPI,self).__init__()
        torch.backends.cudnn.enabled = False
        self.batch_size = args['batch_size']
        self.type = args['task_type']
        self.embedding_size = args['emb_dim']
        self.drop = args['dropout']
        self.output_dim = args['output_dim']
        # gcn
        self.gcn1 = GATConv(self.embedding_size,self.embedding_size,3)
        self.gcn2 = GATConv(self.embedding_size*3,self.embedding_size*3,3)
        self.gcn3 = GATConv(self.embedding_size*9,self.embedding_size*9,1)
        self.relu = nn.ReLU()
        self.fc_g1 = torch.nn.Linear(self.embedding_size*9, self.output_dim)

        self.maxpooling = MaxPooling()
        self.avgpooling = AvgPooling()
        self.dropout = nn.Dropout(self.drop)

        #textcnn
        self.textcnn = ConvsLayer(self.embedding_size)
        self.textflatten = nn.Linear(128,self.output_dim)
        # combined layers
        self.w1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.fc1 = nn.Linear(self.output_dim*2, 512)
        self.fc2 = nn.Linear(512,256)
        self.out = nn.Linear(256, 1)

    # input1 input2
    def forward(self,G1,pad_dmap1,G2,pad_dmap2):
        # protein1
        g1 = self.relu(self.gcn1(G1,G1.ndata['feat']))
        g1 = g1.reshape(-1,self.embedding_size*3)
        g1 = self.relu(self.gcn2(G1, g1))
        g1 = g1.reshape(-1,self.embedding_size*9)
        g1 = self.relu(self.gcn3(G1, g1))
        g1 = g1.reshape(-1,self.embedding_size*9)
        G1.ndata['feat']=g1
        g1_maxpooling = self.maxpooling(G1,G1.ndata['feat'])  
        # flatten
        g1 = self.relu(self.fc_g1(g1_maxpooling))

        seq1 = self.textcnn(pad_dmap1)
        seq1 = self.relu(self.textflatten(seq1))
        # combine g1 and pic1 
        w1 = F.sigmoid(self.w1)
        gc1 = torch.add((1-w1)*g1,w1*seq1) 

        #protein2
        g2 = F.relu(self.gcn1(G2,G2.ndata['feat']))
        g2 = g2.reshape(-1,self.embedding_size*3)
        #g2 = self.n1(g2)
        g2 = F.relu(self.gcn2(G2, g2))
        g2 = g2.reshape(-1,self.embedding_size*9)
        #g2 = self.n2(g2)
        g2 = F.relu(self.gcn3(G2, g2))
        g2 = g2.reshape(-1,self.embedding_size*9)
        #g2 = self.n3(g2)
        G2.ndata['feat']=g2
        g2_maxpooling = self.maxpooling(G2,G2.ndata['feat'])
        # flatten
        g2 = self.relu(self.fc_g1(g2_maxpooling))

        seq2 = self.textcnn(pad_dmap2)
        seq2 = self.relu(self.textflatten(seq2))
        # combine g1 and pic1 
        gc2 = torch.add((1-w1)*g2,w1*seq2)
        #gc2 = torch.add(g2,pic2)   

        # combine gc1 and gc2
        gc = torch.cat([gc1,gc2],dim=1) 
        # add some dense layers
        gc = self.fc1(gc)
        gc = self.relu(gc)
        gc = self.dropout(gc)
        gc = self.fc2(gc)
        gc = self.relu(gc)
        gc = self.dropout(gc)
        out = self.out(gc)
        output = F.sigmoid(out)
        return output