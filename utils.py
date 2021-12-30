import torch
import re
import numpy as np
from selfies import encoder as smiles2selfies
from selfies import decoder as selfies2smiles

def getDataSet(FoldPath):
    with open(FoldPath, 'r') as f:
        Cpi_list = f.read().strip().split('\n')
    """Exclude data contains '.' in the SMILES format."""
    Cpi_list = [d for d in Cpi_list if '.' not in d.strip().split()[0]]
    DataSet = [cpi.strip().split() for cpi in Cpi_list]
    return DataSet#[[smiles, sequence, interaction],.....]

# Create necessary smileses, proteins/contact maps, lengths, and target
def make_variables(lines, proteins, properties, letters, seq2numDic, contactPath):
    lines = get_selfies_list(lines)
    sequence_and_length = [line2voc_arr(line,letters) for line in lines]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
    contactMaps_and_sizes = [protein2contact_arr(str(seq2numDic[protein.encode('utf-8')]), contactPath) for protein in proteins]
    contactMaps = [cm[0] for cm in contactMaps_and_sizes]
    contact_sizes = torch.LongTensor([cm[1] for cm in contactMaps_and_sizes])
    return pad_sequences(vectorized_seqs, seq_lengths, contactMaps, contact_sizes, properties) 
    
def protein2contact_arr(protein_num, contactPath):
    contactMap = np.load(contactPath + protein_num + '.npz')['contact']
    return contactMap, contactMap.shape[0]

def line2voc_arr(line,letters):
    arr = []
    regex = '(\[[^\[\]]{1,10}\])'
    char_list = re.split(regex, line)
    for li, char in enumerate(char_list):
        if char.startswith('['):
               arr.append(letterToIndex(char,letters)) 
        else:
            chars = [unit for unit in char]

            for i, unit in enumerate(chars):
                arr.append(letterToIndex(unit,letters))
    return arr, len(arr)

def letterToIndex(letter,letters):
    return letters.index(letter)

# pad sequences and sort the tensor
def pad_sequences(vectorized_seqs, seq_lengths, contactMaps, contact_sizes, properties):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    contactMaps_tensor = torch.zeros((len(contactMaps), contact_sizes.max(), contact_sizes.max())).float()
    # contactMaps_tensor = torch.ones((len(contactMaps), contact_sizes.max(), contact_sizes.max())).float()*(-1.0)
    
    for idx, (con, con_size) in enumerate(zip(contactMaps, contact_sizes)):
        contactMaps_tensor[idx, :con_size, :con_size] = torch.FloatTensor(con)

    # Sort tensors by their length
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    contactMaps_tensor = contactMaps_tensor[perm_idx]
    contact_sizes = contact_sizes[perm_idx]

    # Also sort the target, contactmap in the same order
    target = properties.double()
    if len(properties):
        target = target[perm_idx]

    contactMaps_tensor = contactMaps_tensor.unsqueeze(1)  # [batchsize,1,max_length,max_length]
    # print(contactMaps_tensor.size())
    # Return variables
    # DataParallel requires everything to be a Variable
    return seq_tensor, seq_lengths, contactMaps_tensor, contact_sizes, target


def get_selfies_list(smiles_list):
    selfies_list = list()
    for smiles in smiles_list:
        selfies_list.append(smiles2selfies(smiles))
    return selfies_list

def getLetters(path):
    with open(path, 'r') as f:
        chars = f.read().split()
    return chars