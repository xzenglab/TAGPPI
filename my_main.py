import torch
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

import sys
from my_train_and_validation import *

def main():
    
    if len(sys.argv) > 1:
        datasetname, rst_file, pkl_path,batchsize = sys.argv[1:]
        batchsize = int(batchsize)
    else: 
        datasetname = 'yeast'
        rst_file = './results/yeast_pipr.tsv'
        pkl_path = './model_pkl/GAT'
        batchsize = 64
    #losses,accs,testResults = train(trainArgs)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train(trainArgs)

if __name__ == "__main__":
    main()