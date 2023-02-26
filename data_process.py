import json
import pickle
from collections import OrderedDict
from itertools import chain
from pathlib import Path

import networkx as nx
import numpy as np
import typer
from rdkit import Chem
from tqdm import tqdm

import constants
from utils import DTADataset, Graph


def dic_normalize(dic: dict[str, float]) -> dict[str, float]:
    # get min and max values according to values of passed dictionary
    max_value = dic[max(dic, key=dic.get)]  # type: ignore
    min_value = dic[min(dic, key=dic.get)]  # type: ignore
    interval = float(max_value) - float(min_value)
    # loop through all items in dict and normalizing its values
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    # adding new item into dictionary
    dic["X"] = (max_value + min_value) / 2.0
    return dic


def residue_features(norm_res_prop: list[dict], residue: str) -> np.ndarray:
    res_property1 = [
        1 if residue in list else 0 for list in constants.protien_properties
    ]
    res_property2 = [dict[residue] for dict in norm_res_prop]
    res_property = np.array(res_property1 + res_property2)
    return res_property


def one_of_k_encoding(x: str, allowable_set: list, unk=False) -> list[bool]:
    if x not in allowable_set:
        if not unk:
            raise Exception(f"input {x} not in allowable set{allowable_set}")

        # Maps inputs not in the allowable set to the last element in allowable_set.
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


# mol atom feature for mol graph
def atom_features(atom: Chem.rdchem.Atom) -> np.ndarray:
    atom_sym = one_of_k_encoding(atom.GetSymbol(), constants.atom_symbols, unk=True)
    atom_deg = one_of_k_encoding(atom.GetDegree(), list(range(11)))
    atom_total_num_hs = one_of_k_encoding(
        atom.GetTotalNumHs(), list(range(11)), unk=True
    )
    atom_implicit_valence = one_of_k_encoding(
        atom.GetImplicitValence(), list(range(11)), unk=True
    )
    atom_is_aromatic = atom.GetIsAromatic()
    atom_features = list(
        chain(
            atom_sym,
            atom_deg,
            atom_total_num_hs,
            atom_implicit_valence,
            [atom_is_aromatic],
        )
    )
    # return a boolean list with length equals to (44 +11 +11 +11 +1) represents atom features
    return np.array(atom_features)


# mol smile to mol graph edge index
def smile_to_graph(smile: str) -> Graph:
    mol = Chem.MolFromSmiles(smile)  # type: ignore
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)  # get boolean list
        # divide each item in list by its length, true = 1 , false = 0
        # so this values (feature / sum(feature)) represrnts number of true relative to false
        # features list represents whole compound (collection of atoms)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:  # type: ignore
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))  # there is self edge

    # NOTE: why 0.5 while the matrix contain either 0 or 1
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    smile_graph = Graph(np.array(c_size), np.array(features), np.array(edge_index))
    return smile_graph


def PSSM_calculation(aln_file: Path, pro_seq: str) -> np.ndarray:
    pfm_mat = np.zeros((len(constants.pro_res_list), len(pro_seq)))
    with open(aln_file, "r") as f:
        line_count = len(f.readlines())
        for line in f.readlines():
            if len(line) != len(pro_seq):
                print("error", len(line), len(pro_seq))
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


def seq_feature(norm_res_prop: list[dict], pro_seq: str) -> np.ndarray:
    pro_hot = np.zeros((len(pro_seq), len(constants.pro_res_list)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        pro_hot[i] = one_of_k_encoding(pro_seq[i], constants.pro_res_list)
        pro_property[i] = residue_features(norm_res_prop, pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)


def target_to_feature(
    norm_res_prop: list[dict], target_key: str, target_sequence: str, aln_dir: Path
) -> list:
    aln_file = aln_dir.joinpath(f"{target_key}.aln")
    pssm = PSSM_calculation(aln_file, target_sequence)
    other_feature = seq_feature(norm_res_prop, target_sequence)
    feature = np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)
    return feature.tolist()


def target_to_graph(dataset_path: Path, target_key: str, target_sequence: str) -> Graph:
    aln_dir = dataset_path.joinpath("aln")
    contact_dir = dataset_path.joinpath("pconsc4")

    target_edge_index = []
    target_size = len(target_sequence)
    contact_file = contact_dir.joinpath(f"{target_key}.npy")
    contact_map = np.load(contact_file)
    contact_map += np.matrix(np.eye(contact_map.shape[0]))
    index_row, index_col = np.where(contact_map >= 0.5)
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])

    # normalizing all properties values
    norm_res_prop = [
        dic_normalize(re_prop) for re_prop in constants.residual_properties
    ]

    target_feature = target_to_feature(
        norm_res_prop, target_key, target_sequence, aln_dir
    )
    target_graph = Graph(
        np.array(target_size), np.array(target_feature), np.array(target_edge_index)
    )
    return target_graph


# to judge whether the required files exist
def valid_target(file_name: str, dataset_path: Path) -> bool:
    contact_file = dataset_path.joinpath(f"pconsc4/{file_name}.npy")
    aln_file = dataset_path.joinpath(f"aln/{file_name}.aln")
    return all([contact_file.exists(), aln_file.exists()])


def validate(
    dataset_path: Path,
    drugs: list,
    proteins_dict: OrderedDict,
    affinity: np.ndarray,
    folds: list,
) -> tuple:
    """get combinations of drugs and proteins with affinity strength when meet with each other"""
    prot_keys = list(proteins_dict.keys())
    rows, cols = np.where(np.isnan(affinity) == False)
    rows, cols = rows[folds], cols[folds]

    drugs_list = []
    prot_names = []
    affinities = []

    for pair_ind in range(len(rows)):
        # ensure the contact and aln files exists
        if not valid_target(prot_keys[cols[pair_ind]], dataset_path):
            continue
        drugs_list.append(drugs[rows[pair_ind]])
        prot_names.append(prot_keys[cols[pair_ind]])
        affinities.append(affinity[rows[pair_ind], cols[pair_ind]])
    return drugs_list, prot_names, affinities


def load_data(
    dataset_path: Path, fold_setting: list
) -> tuple[list, list, list, OrderedDict, dict]:
    dataset_name = dataset_path.stem
    ligands_file = dataset_path.joinpath("ligands.txt")
    proteins_file = dataset_path.joinpath("proteins.txt")
    affinities_file = dataset_path.joinpath("affinities")
    ligands = json.load(open(ligands_file), object_pairs_hook=OrderedDict)
    proteins_dict = json.load(open(proteins_file), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(affinities_file, "rb"), encoding="latin1")

    if dataset_name == "davis":
        affinity = [-np.log10(y / 1e9) for y in affinity]
    mol_drugs = []
    ligands_dict = {}
    for d in tqdm(ligands.keys(), desc=f"loading data from {str(dataset_path)}"):
        ligand = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)  # type: ignore
        ligands_dict[d] = ligand
        mol_drugs.append(ligand)

    affinity = np.asarray(affinity)
    drugs, prots, affinities = validate(
        dataset_path, mol_drugs, proteins_dict, affinity, fold_setting
    )
    return drugs, prots, affinities, proteins_dict, ligands_dict


def create_data(dataset_path: Path, fold_setting: list) -> DTADataset:
    smiles, prot_names, affinity, proteins_dict, ligands_dict = load_data(
        dataset_path, fold_setting
    )
    smile_graph_dict = {}
    for smile in tqdm(ligands_dict.values(), desc="converting smiles to graphs"):
        smile_graph_dict[smile] = smile_to_graph(smile)

    target_graph_dict = {}
    for key in tqdm(proteins_dict.keys(), desc="converting proteins to graphs"):
        target_graph_dict[key] = target_to_graph(dataset_path, key, proteins_dict[key])

    dataset = DTADataset(
        root=str(dataset_path),
        drugs=np.array(smiles),
        target_key=np.array(prot_names),
        y=np.array(affinity),
        smile_graph_dict=smile_graph_dict,
        target_graph_dict=target_graph_dict,
    )  # type: ignore
    return dataset


def create_test_data(dataset_path: Path) -> DTADataset:
    test_file = open(dataset_path.joinpath("folds/test_fold_setting1.txt"))
    fold_setting = json.load(test_file)
    test_data = create_data(dataset_path, fold_setting)
    return test_data


def create_train_data(
    dataset_path: Path, fold: int = 0
) -> tuple[DTADataset, DTADataset]:
    train_file = dataset_path.joinpath("folds/train_fold_setting1.txt")
    train_fold_origin = json.load(open(train_file))
    valid_fold = train_fold_origin.pop(fold)  # get one element
    train_folds = list(chain(*train_fold_origin))  # get other elements
    train_data = create_data(dataset_path, train_folds)
    valid_data = create_data(dataset_path, valid_fold)
    return train_data, valid_data


def main(fold_number: int = typer.Option(..., prompt=True)):
    for data in ["davis", "kiba"]:
        dataset_path = constants.project_path.joinpath(f"data/{data}")
        train_dataset, valid_dataset = create_train_data(dataset_path, fold_number)
        test_dataset = create_test_data(dataset_path)
        train_dataset.save(fold_number, "train")
        valid_dataset.save(fold_number, "valid")
        test_dataset.save(fold_number, "test")


if __name__ == "__main__":
    typer.run(main)
