import Bio.PDB
import numpy as np

def get_center_atom(residue):
    if residue.has_id('CA'):
        c_atom = 'CA'
    elif residue.has_id('N'):
        c_atom = 'N'
    elif residue.has_id('C'):
        c_atom = 'C'
    elif residue.has_id('O'):
        c_atom = 'O'
    elif residue.has_id('CB'):
        c_atom = 'CB'
    elif residue.has_id('CD'):
        c_atom = 'CD'
    else:
        c_atom = 'CG'
    return c_atom


def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""

    c_atom1 = get_center_atom(residue_one)
    c_atom2 = get_center_atom(residue_two)
    diff_vector  = residue_one[c_atom1].coord - residue_two[c_atom2].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))

def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    residue_len = 0
    for row, residue_one in enumerate(chain_one):
        hetfield = residue_one.get_id()[0]
        hetname = residue_one.get_resname()
        if hetfield == " " and hetname in aa_codes.keys():
            residue_len = residue_len + 1
    answer = np.zeros((residue_len, residue_len), np.float)
    x = -1
    for residue_one in chain_one:
        y = -1
        hetfield1 = residue_one.get_id()[0]
        hetname1 = residue_one.get_resname()
        if hetfield1 == ' ' and hetname1 in aa_codes.keys():
            x = x + 1
            for residue_two in chain_two:
                hetfield2 = residue_two.get_id()[0]
                hetname2 = residue_two.get_resname()
                if hetfield2 == ' ' and hetname2 in aa_codes.keys():
                    y = y + 1
                    answer[x, y] = calc_residue_dist(residue_one, residue_two)
    for i in range(residue_len):
        answer[i,i] = 100
    return answer


def calc_contact_map(pdb_id,chain_id):
    pdb_path = data_root_path + pdb_id + '.pdb'
    structure = Bio.PDB.PDBParser().get_structure(pdb_id, pdb_path)
    model = structure[0]
    dist_matrix = calc_dist_matrix(model[chain_id], model[chain_id])
    contact_map = (dist_matrix < 8.0).astype(np.int)
    #print('contact map shape:',contact_map.shape)
    return contact_map

aa_codes = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
    'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TYR': 'Y', 'TRP': 'W',
}

# for example
sequence = 'MQIVMFDRQSIFIHGMKISLQQRIPGVSIQGASQADELWQKL'
data_root_path = 'your_data_root_path'
pdb_id = 'pad_file_name'
contact_map = calc_contact_map(pdb_id,'A')
contact_file = 'save_path.npz'
np.savez(contact_file,seq = sequence, contact = contact_map)

# contact = np.load(contact_file)
# print(contact.files)
# print(contact['seq'])
# print(contact['contact'])
