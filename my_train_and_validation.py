from sklearn import metrics
import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
import  numpy as np
from graph_cmap_loader import * 

from my_args import *
import dgl 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.metrics import confusion_matrix

device = torch.device('cuda')

def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx,(G1,dmap1,G2,dmap2,y) in enumerate(loader):
            pad_dmap1 = pad_dmap(dmap1)
            pad_dmap2 = pad_dmap(dmap2)  
            output = model(dgl.batch(G1),pad_dmap1,dgl.batch(G2),pad_dmap2)
            output = torch.round(output.squeeze(1))
            total_preds = torch.cat((total_preds.cpu(), output.cpu()), 0)
            total_labels = torch.cat((total_labels.cpu(), y.float().cpu()), 0)
            
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

def train(trainArgs):

    train_losses = []
    train_accs = []

    for i in range(trainArgs['epochs']): 
        print("Running EPOCH",i+1)
        total_loss = 0
        n_batches = 0
        correct = 0
        train_loader = trainArgs['train_loader']
        optimizer = trainArgs['optimizer']
        criterion = trainArgs["criterion"]
        attention_model = trainArgs['model']

        for batch_idx,(G1,dmap1,G2,dmap2,y) in enumerate(train_loader):  

            y_pred = attention_model(dgl.batch(G1),pad_dmap(dmap1),dgl.batch(G2), pad_dmap(dmap2))
            correct+=torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)),y.type(torch.DoubleTensor)).data.sum()
            loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),y.type(torch.DoubleTensor))
            total_loss+=loss.data
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            n_batches+=1
                
        avg_loss = total_loss/n_batches
        acc = correct.numpy()/(len(train_loader.dataset))
        
        train_losses.append(avg_loss)
        train_accs.append(acc)
        
        print("train avg_loss is",avg_loss)
        print("train ACC = ",acc)
        
        if(trainArgs['doSave']):
            torch.save(attention_model.state_dict(), pkl_path+'epoch'+'%d.pkl'%(i+1))
        # test
        total_labels,total_preds = predicting(attention_model,device,test_loader)
        test_acc = accuracy_score(total_labels, total_preds)
        test_prec = precision_score(total_labels, total_preds)
        test_recall = recall_score(total_labels, total_preds)
        test_f1 = f1_score(total_labels, total_preds)
        test_auc = roc_auc_score(total_labels, total_preds)
        con_matrix = confusion_matrix(total_labels, total_preds)
        test_spec = con_matrix[0][0]/(con_matrix[0][0]+con_matrix[0][1])
        test_mcc = (con_matrix[0][0]*con_matrix[1][1]-con_matrix[0][1]*con_matrix[1][0])/(((con_matrix[1][1]+con_matrix[0][1])*(con_matrix[1][1]+con_matrix[1][0])*(con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0]+con_matrix[1][0]))**0.5)
        print("acc: ",test_acc," ; prec: ",test_prec," ; recall: ",test_recall," ; f1: ",test_f1," ; auc: ",test_auc," ; spec:",test_spec," ; mcc: ",test_mcc)
        with open(rst_file, 'a+') as fp:
            fp.write('epoch:' + str(i+1) + '\ttrainacc=' + str(acc) +'\ttrainloss=' + str(avg_loss.item()) +'\tacc=' + str(test_acc) + '\tprec=' + str(test_prec) + '\trecall=' + str(test_recall) +  '\tf1=' + str(test_f1) + '\tauc=' + str(test_auc) + '\tspec='+str(test_spec)+ '\tmcc='+str(test_mcc)+'\n')
    