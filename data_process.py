import json
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem

import constants
from utils import DTADataset

@dataclass
class Smile_graph():
    c_size: int
    features: list
    edge_index: list

@dataclass
class Target_graph():
    target_size: int
    target_feature: np.ndarray
    target_edge_index: np.ndarray    
    
# nomarlize
def dic_normalize(dic: dict[str, float]) -> dict[str, float]: # no side effect
    # get min and max values according to values of passed dictionary
    max_value = dic[max(dic, key=dic.get)] # type: ignore
    min_value = dic[min(dic, key=dic.get)] # type: ignore
    interval = float(max_value) - float(min_value)
    # loop through all items in dict and normalizing its values
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    # adding new item into dictionary
    dic['X'] = (max_value + min_value) / 2.0
    return dic

# get features of passes residue
def residue_features(normalized_residual_properties: list[dict], residue: str) -> np.ndarray:
    res_property1 = [1 if residue in list else 0 for list in constants.protien_properties]
    res_property2 = [dict[residue] for dict in normalized_residual_properties]
    res_property = np.array(res_property1 + res_property2)
    return res_property

# one ont encoding
def one_of_k_encoding(x: str, allowable_set: list, unk=False) -> list[bool]: # no side effect
    if x not in allowable_set:
        if not unk:
            raise Exception(f'input {x} not in allowable set{allowable_set}')

        # Maps inputs not in the allowable set to the last element in allowable_set.
        x = allowable_set[-1]
    return [x==s for s in allowable_set]

# mol atom feature for mol graph
def atom_features(atom: Chem.rdchem.Atom) -> np.ndarray: # no side effect, depends on another function [one_of_k_encoding]
    atom_features = one_of_k_encoding(atom.GetSymbol(),constants.atom_symbols, unk=True) \
        + one_of_k_encoding(atom.GetDegree(), list(range(11))) \
        + one_of_k_encoding(atom.GetTotalNumHs(), list(range(11)), unk=True) \
        + one_of_k_encoding(atom.GetImplicitValence(), list(range(11)), unk=True) \
        + [atom.GetIsAromatic()]
    # return a boolean list with length equals to (44 +11 +11 +11 +1) represents atom features
    return np.array(atom_features)


# mol smile to mol graph edge index
def smile_to_graph(smile:str) -> Smile_graph: # no side effect, depends on another function [atom_features]
    mol = Chem.MolFromSmiles(smile) # type: ignore    
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom) # get boolean list
        # divide each item in list by its length, true = 1 , false = 0
        # so this values (feature / sum(feature)) represrnts number of true relative to false
        features.append(feature / sum(feature)) # features list represents whole compound (collection of atoms)

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges: # type: ignore
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))  # there is self edge

    # NOTE: why 0.5 while the matrix contain either 0 or 1
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    smile_graph = Smile_graph(c_size, features, edge_index)
    return smile_graph


# target feature for target graph
def PSSM_calculation(aln_file: Path, pro_seq: str) -> np.ndarray: 
    pfm_mat = np.zeros((len(constants.pro_res_list), len(pro_seq)))
    with open(aln_file, 'r') as f:
        line_count = len(f.readlines())
        for line in f.readlines():
            if len(line) != len(pro_seq):
                print('error', len(line), len(pro_seq))
                continue
            count = 0
            for res in line:
                if res not in constants.pro_res_list:
                    count += 1
                    continue
                pfm_mat[constants.pro_res_list.index(res), count] += 1
                count += 1
    pseudocount = 0.8
    pssm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
    return pssm_mat

def seq_feature(normalized_residual_properties: list[dict], pro_seq: str) -> np.ndarray:
    pro_hot = np.zeros((len(pro_seq), len(constants.pro_res_list)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        pro_hot[i] = one_of_k_encoding(pro_seq[i], constants.pro_res_list)
        pro_property[i] = residue_features(normalized_residual_properties, pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)

def target_to_feature(normalized_residual_properties: list[dict], target_key: str, target_sequence: str, aln_dir:Path) -> np.ndarray:
    # no side effect, depends on another functions [PSSM_calculation, seq_feature]
    aln_file = aln_dir.joinpath(f"{target_key}.aln")
    pssm = PSSM_calculation(aln_file, target_sequence)
    other_feature = seq_feature(normalized_residual_properties, target_sequence)
    feature =  np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)
    return feature

# pconsc4 predicted contact map save in data/dataset_name/pconsc4
def target_to_graph(dataset_path: Path, target_key: str, target_sequence: str) -> Target_graph:
    aln_dir = dataset_path.joinpath('aln')
    contact_dir = dataset_path.joinpath('pconsc4')

    target_edge_index = []
    target_size = len(target_sequence)
    contact_file = contact_dir.joinpath(f"{target_key}.npy")
    contact_map = np.load(contact_file)
    contact_map += np.matrix(np.eye(contact_map.shape[0]))
    index_row, index_col = np.where(contact_map >= 0.5)
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    
    # normalizing all properties values
    normalized_residual_properties = [dic_normalize(re_prop) for re_prop in constants.residual_properties]

    target_feature = target_to_feature(normalized_residual_properties, target_key, target_sequence, aln_dir)
    target_edge_index = np.array(target_edge_index)
    target_graph = Target_graph(target_size, target_feature, target_edge_index)
    return target_graph


# to judge whether the required files exist
def valid_target(file_name: str, dataset_path:Path) -> bool: # no side effect
    contact_file = dataset_path.joinpath(f"pconsc4/{file_name}.npy")
    aln_file = dataset_path.joinpath(f"aln/{file_name}.aln")
    return all([contact_file.exists(), aln_file.exists()])


def data_to_csv(csv_file: Path, data_list: list) -> bool: # no side effect
    try:
        with open(csv_file, 'w') as f:
            f.write('compound_iso_smiles,target_sequence,target_key,affinity\n')
            for data in data_list:
                f.write(','.join(map(str, data)) + '\n')
        return True
    except Exception:
        return False
    
def load_data(dataset_path: Path) -> tuple[list, np.ndarray, OrderedDict]:
    print("start loading ....")
    dataset_name = dataset_path.stem
    ligands = json.load(open(dataset_path.joinpath('ligands_can.txt')),object_pairs_hook=OrderedDict)
    proteins = json.load(open(dataset_path.joinpath('proteins.txt')),object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(dataset_path.joinpath('Y'), 'rb'), encoding='latin1')

    mol_drugs = []
    drug_smiles = []
    # smiles
    for d in ligands.keys():
        ligand = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True) # type: ignore
        mol_drugs.append(ligand)
        drug_smiles.append(ligands[d])
    if dataset_name == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)
    print("end loading ....")

    return mol_drugs, affinity, proteins

def save_data(csv_file: Path, drugs: list, proteins: OrderedDict, affinity: np.ndarray, folds: list) -> bool:
    # no side effect, depends on another functions [valid_target, data_to_csv]
    dataset_path = csv_file.parent
    prots = list(proteins.values())
    prot_keys = list(proteins.keys())

    rows, cols = np.where(np.isnan(affinity) == False)
    rows, cols = rows[folds], cols[folds]
    data = []
    for pair_ind in range(len(rows)):
        # ensure the contact and aln files exists
        if not valid_target(prot_keys[cols[pair_ind]], dataset_path):
            continue
        ls = [drugs[rows[pair_ind]], prots[cols[pair_ind]], prot_keys[cols[pair_ind]], affinity[rows[pair_ind], cols[pair_ind]]]
        data.append(ls)

    saved = data_to_csv(csv_file, data)
    return saved

def create_data(dataset_path: Path, fold_setting:list, csv_file_path: Path) -> DTADataset:
    # no side effect, depends on another functions [load_data, save_data, target_to_graph, smile_to_graph] 
    train_or_test = "train" if "train" in csv_file_path.stem else "test"
    dataset_name = f"{dataset_path.stem}_{train_or_test}"

    print(f"start loading {dataset_name}....")
    mol_drugs, affinity, proteins = load_data(dataset_path)
    print(f"end loading {dataset_name}....")

    prot_keys = list(proteins.keys())

    print(f"start saving {dataset_name}....")
    saved = save_data(csv_file_path, mol_drugs, proteins, affinity, fold_setting)
    print(f"end saving {dataset_name}....")

    print(f"start processing {dataset_name}....")
    smile_graph = {smile : vars(smile_to_graph(smile)).values() for smile in mol_drugs}
    target_graph = {key : vars(target_to_graph(dataset_path, key, proteins[key])).values() for key in prot_keys if valid_target(key, dataset_path)}
    print(f"end processing {dataset_name}....")
    if saved:
        df = pd.read_csv(csv_file_path)
        drugs= np.asarray(list(df['compound_iso_smiles']))
        prot_keys = np.asarray(list(df['target_key']))
        Y = np.asarray(list(df['affinity']))
        dataset = DTADataset(root=str(dataset_path), dataset_name=dataset_name, xd=drugs, y=Y,target_key=prot_keys, smile_graph=smile_graph, target_graph=target_graph)
        return dataset
    else:
        raise ValueError("the data are not saved")

def create_test_data(dataset_path: Path) -> DTADataset:
    # no side effect, depends on another function [create_data]
    fold_setting = json.load(open(dataset_path.joinpath('folds/test_fold_setting1.txt')))
    dataset_name = dataset_path.stem
    csv_file_path = dataset_path.joinpath(f"{dataset_name}_test.csv")
    test_data = create_data(dataset_path, fold_setting, csv_file_path)
    return test_data

def create_train_data(dataset_path: Path, fold: int=0) -> tuple[DTADataset, DTADataset]:
    # no side effect, depends on another function [create_data]
    dataset_name = dataset_path.stem
    train_fold_origin = json.load(open(dataset_path.joinpath('folds/train_fold_setting1.txt')))
    valid_fold = train_fold_origin.pop(fold)  # get one element
    train_folds = np.array(train_fold_origin).flatten().tolist()  # get other elements

    train_csv_file = dataset_path.joinpath(f"{dataset_name}_fold_{str(fold)}_train.csv")
    valid_csv_file = dataset_path.joinpath(f"{dataset_name}_fold_{str(fold)}_valid.csv")
    train_data = create_data(dataset_path, train_folds, train_csv_file)
    valid_data = create_data(dataset_path, valid_fold, valid_csv_file)
    return train_data, valid_data
if __name__ == "__main__":
    for data in ["davis"]:
        dataset_path = constants.project_path.joinpath(f"data/{data}")
        train_dataset, valid_dataset = create_train_data(dataset_path)
        test_data = create_test_data(dataset_path)

