import torch
import dgl
import scipy.sparse as spp
from seq2tensor import s2t 
import os
import numpy as np
import re
import sys
from torch.utils.data import DataLoader,Dataset
import sys
from my_main import *

if len(sys.argv) > 1:
    datasetname, rst_file, pkl_path, batchsize = sys.argv[1:]
    batchsize = int(batchsize)
else: 
    datasetname = 'yeast'
    rst_file = './results/yeast_pipr.tsv'
    pkl_path = './model_pkl/GAT'
    batchsize = 64

device = torch.device('cuda')

def collate(samples):

    graphs1,dmaps1,graphs2,dmaps2,labels = map(list, zip(*samples))
    return graphs1,dmaps1,graphs2,dmaps2,torch.tensor(labels)

cmaproot = './data/'+datasetname+'/real_cmap/'
embed_data = np.load("./data/"+datasetname+"/dictionary/protein_embeddings.npz")

def default_loader(cpath,pid):

    cmap_data = np.load(cpath)
    nodenum = len(str(cmap_data['seq']))
    cmap = cmap_data['contact']
    g_embed = torch.tensor(embed_data[pid][:nodenum]).float().to(device)

    adj = spp.coo_matrix(cmap)
    G = dgl.DGLGraph(adj).to(device)
    G = G.to(torch.device('cuda'))
    G.ndata['feat'] = g_embed
    
    if nodenum > 1000:
        textembed = embed_data[pid][:1000]
    elif nodenum < 1000:
        textembed = np.concatenate((embed_data[pid], np.zeros((1000 - nodenum, 1024))))
    
    textembed = torch.tensor(textembed).float().to(device)
    return G,textembed


class MyDataset(Dataset): 

    def __init__(self,type,transform=None,target_transform=None, loader=default_loader):
        
        super(MyDataset,self).__init__()
        pns=[]
        with open('./data/'+datasetname+'/actions/'+type+'_cmap.actions.tsv', 'r') as fh: 	        
            for line in fh: 
                line = line.strip('\n')
                line = line.rstrip('\n')
                words = re.split('  |\t',line)
                pns.append((words[0],words[1],int(words[2])))
                
        self.pns = pns
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader        

    def __getitem__(self, index):
        p1,p2, label = self.pns[index]
        G1,embed1 = self.loader(cmaproot+p1+'.npz',p1) 
        G2,embed2 = self.loader(cmaproot+p2+'.npz',p2)
        return G1,embed1,G2,embed2,label  

       
    def __len__(self):
        return len(self.pns)

def pad_sequences(vectorized_seqs, seq_lengths, contactMaps, contact_sizes, properties):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    contactMaps_tensor = torch.zeros((len(contactMaps), contact_sizes.max(), contact_sizes.max())).float()
    # contactMaps_tensor = torch.ones((len(contactMaps), contact_sizes.max(), contact_sizes.max())).float()*(-1.0)
    
    for idx, (con, con_size) in enumerate(zip(contactMaps, contact_sizes)):
        contactMaps_tensor[idx, :con_size, :con_size] = torch.FloatTensor(con)

    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    contactMaps_tensor = contactMaps_tensor[perm_idx]
    contact_sizes = contact_sizes[perm_idx]

    target = properties.double()
    if len(properties):
        target = target[perm_idx]

    contactMaps_tensor = contactMaps_tensor.unsqueeze(1)  # [batchsize,1,max_length,max_length]
    return seq_tensor, seq_lengths, contactMaps_tensor, contact_sizes, target

def pad_dmap(dmaplist):
   
    pad_dmap_tensors = torch.zeros((len(dmaplist), 1000, 1024)).float()
    for idx, d in enumerate(dmaplist):
        d = d.float().cpu()
        pad_dmap_tensors[idx] = torch.FloatTensor(d)
    pad_dmap_tensors = pad_dmap_tensors.unsqueeze(1).cuda()
    return pad_dmap_tensors

train_dataset = MyDataset(type = 'train')
train_loader = DataLoader(dataset = train_dataset, batch_size = batchsize, shuffle=True,drop_last = True,collate_fn=collate)
test_dataset = MyDataset(type = 'test')
test_loader = DataLoader(dataset = test_dataset, batch_size = batchsize , shuffle=True,drop_last = True,collate_fn=collate)

