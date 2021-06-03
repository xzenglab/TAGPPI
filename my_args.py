from graph_cmap_loader import *
from TAGlayer import *

modelArgs = {}
modelArgs['batch_size'] = 8
modelArgs['dropout'] = 0.5
modelArgs['emb_dim'] = 1024
modelArgs['output_dim'] = 128
modelArgs['dense_hid'] = 64
modelArgs['task_type'] = 0
modelArgs['n_classes'] = 1


trainArgs = {}
trainArgs['model'] = GATPPI(modelArgs).cuda()
trainArgs['epochs'] = 200
trainArgs['lr'] = 0.001
trainArgs['train_loader'] = train_loader
trainArgs['doTest'] = True
trainArgs['criterion'] = torch.nn.BCELoss()
trainArgs['optimizer'] = torch.optim.Adam(trainArgs['model'].parameters(),lr=trainArgs['lr'])
trainArgs['lr_scheduler'] = torch.optim.lr_scheduler.StepLR(trainArgs['optimizer'], step_size = 20, gamma = 0.1, last_epoch=-1)
trainArgs['doSave'] = False
