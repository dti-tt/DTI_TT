# downstream
import pandas as pd
import numpy as np
from multiprocessing import Pool
import dgl.backend as F
from dgl.data.utils import save_graphs
from dgllife.utils.io import pmap
from rdkit import Chem
from scipy import sparse as sp
import argparse 

from KPGT.src.data.featurizer import smiles_to_graph_tune
from KPGT.src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized

# KPGT_check
from KPGT.src.utils import set_random_seed
from KPGT.src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from KPGT.src.model_config import config_dict
from KPGT.src.data.collator import Collator_tune
from KPGT.src.data.finetune_dataset import MoleculeDataset
from KPGT.src.model.light import LiGhTPredictor as LiGhT

import torch
from torch.utils.data import DataLoader

class KPGTProcessor:
    def __init__(self, drug_unique_csv_path, task_names):
        self.drug_unique_csv_path = drug_unique_csv_path
        self.path_length = 5 # default
        self.save_path = './usage'
        self.task_names = task_names # list
        self.n_jobs = 32 # default
        self.dataset = 'unmatched_drug_unique'
        # './usage/drug_unique.csv'
        
    # downstream
    def preprocess_downstream_dataset(self):
        df = pd.read_csv(self.drug_unique_csv_path)
        # print(df.head())
        
        # save_path 1
        cache_file_path = f"{self.save_path}/{self.dataset}_{self.path_length}.pkl"
        smiless = df.SMILES.values.tolist()
        # print(smiless)
        
        graphs = pmap(smiles_to_graph_tune,
                                        smiless,
                                        max_length = self.path_length,
                                        n_virtual_nodes=2,
                                        n_jobs=self.n_jobs)
        valid_ids = []
        valid_graphs = []
        for i, g in enumerate(graphs):
            if g is not None:
                valid_ids.append(i)
                valid_graphs.append(g)
        _label_values = df[self.task_names].values
        labels = F.zerocopy_from_numpy(
            _label_values.astype(np.float32))[valid_ids]
        
        print('saving graphs')
        
        
        save_graphs(cache_file_path, valid_graphs,
                    labels={'labels': labels}) 
        
        
        print('extracting fingerprints')
        FP_list = []
        for smiles in smiless:
            mol = Chem.MolFromSmiles(smiles)
            FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
        FP_arr = np.array(FP_list)
        FP_sp_mat = sp.csc_matrix(FP_arr)
        print('saving fingerprints')
        
        # save path2
        sp.save_npz(f"{self.save_path}/rdkfp1-7_512.npz", FP_sp_mat)
        
        print('extracting molecular descriptors')
        generator = RDKit2DNormalized()
        
        features_map = Pool(self.n_jobs).imap(generator.process, smiless)
        arr = np.array(list(features_map))
        
        # save path3
        np.savez_compressed(f"{self.save_path}/molecular_descriptors.npz",md=arr[:,1:])
                
    
    def get_features(self):
        # get downstream dataset
        self.preprocess_downstream_dataset()
        
        config = config_dict['base']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f" Using device: {device}")
        vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
        collator = Collator_tune(config['path_length'])
        
        root_path = self.save_path
        mol_dataset = MoleculeDataset(root_path=root_path,
                                      dataset=self.dataset,
                                      dataset_type=None,
                                      task_names=self.task_names)
        loader = DataLoader(mol_dataset, batch_size=32, shuffle=False, num_workers=8, drop_last=False, collate_fn=collator)
        
        model = LiGhT(
            d_node_feats=config['d_node_feats'],
            d_edge_feats=config['d_edge_feats'],
            d_g_feats=config['d_g_feats'],
            d_hpath_ratio=config['d_hpath_ratio'],
            n_mol_layers=config['n_mol_layers'],
            path_length=config['path_length'],
            n_heads=config['n_heads'],
            n_ffn_dense_layers=config['n_ffn_dense_layers'],
            input_drop=0,
            attn_drop=0,
            feat_drop=0,
            n_node_types=vocab.vocab_size
        ).to(device)
        
        model_path = "./pretrained/base/base.pth" # dowmload informatio
        state_dict = torch.load(model_path)
        
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
        print("KPGT pretrained model loaded!")
        
        fps_list = []
        
        for batch_idx, batched_data in enumerate(loader):
            # in finetune_dataset MoleculeDataset
            (_, g, ecfp, md, labels) = batched_data
            ecfp = ecfp.to(device)
            md = md.to(device)
            g = g.to(device)
            fps = model.generate_fps(g, ecfp, md)
            fps_list.extend(fps.detach().cpu().numpy().tolist())
        
        # fps = np.array(fps_list)
        
        np.savez_compressed(f"./usage/result/kpgt_features.npz", fps=np.array(fps_list))
        print(f"The extracted features saved.")
        
        final_file_path = "./usage/result/kpgt_features.npz"
        data = np.load(final_file_path)

        fps_array = data['fps']
        # print("fps_array shape:", fps_array.shape)
        
        return fps_array