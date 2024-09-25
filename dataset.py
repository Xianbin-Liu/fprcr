import torch
from rxnfp import create_rxn_Morgan2FP_separately
import pandas as pd
import numpy as np
from pathlib import Path

class fpDataSet(torch.utils.data.Dataset):
    def __init__(self, reactants, products, labels):
        super(fpDataSet, self).__init__()
        self.reactants = reactants
        self.products = products
        self.labels = labels

    def __len__(self):
        return len(self.reactants)
    
    def __getitem__(self, idx):
        # calculate the fingerprint
        rfp, pfp = create_rxn_Morgan2FP_separately(self.reactants[idx], self.products[idx])
        return rfp, pfp, self.labels[idx]
    
    
def col_fn(batch):
    rfps, pfps, labels = zip(*batch)
    
    rfps = np.array(rfps)
    pfps = np.array(pfps)
    labels = np.array(labels)
    
    return torch.tensor(rfps), torch.tensor(pfps), torch.LongTensor(labels)


def create_fpDataset(data_dir):
    train_ds = create_fpDataset_from_file(Path(data_dir, 'train.csv'))
    val_ds = create_fpDataset_from_file(Path(data_dir, 'val.csv'))
    test_ds = create_fpDataset_from_file(Path(data_dir, 'test.csv'))
    return train_ds, val_ds, test_ds


def create_fpDataset_from_file(dataPath):
    df = pd.read_csv(dataPath, header=0, index_col=False)
    reactants = [smi[0] for smi in df[['reactants']].values.tolist()]
    products = [smi[0] for smi in df[['products']].values.tolist()]
    labels = df[['catalyst1_bin', 'solvent1_bin', 'solvent2_bin', 'reagent1_bin', 'reagent2_bin']].values
    labels = [list(map(lambda x: int(x[1:]), label)) for label in labels.tolist() ]
    return fpDataSet(reactants, products, labels)