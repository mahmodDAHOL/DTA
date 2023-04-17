import json
from collections import OrderedDict

import numpy as np
import pytest
from rdkit import Chem

from src.constants import project_path
from src.data_process import *


@pytest.fixture
def data():
    data = {}
    data_path = project_path.joinpath("tests/data/davis")
    proteins_file = data_path.joinpath("proteins.txt")
    proteins_dict = json.load(proteins_file.open(), object_pairs_hook=OrderedDict)
    target_name = next(iter(proteins_dict.keys()))
    target_sequence = next(iter(proteins_dict.values()))

    ligands_file = data_path.joinpath("ligands.txt")
    ligands_dict = json.load(ligands_file.open(), object_pairs_hook=OrderedDict)
    smile = next(iter(ligands_dict.values()))

    aln_file = data_path.joinpath("aln/AAK1.aln")
    contact_file = data_path.joinpath("pconsc4/AAK1.npy")

    data["data_path"] = data_path
    data["target_key"] = target_name
    data["target_sequence"] = target_sequence
    data["smile"] = smile
    data["aln_file"] = aln_file
    data["contact_file"] = contact_file
    return data

def test_dic_normalize():
    # Test that the function correctly normalizes a dictionary
    dic = {"a": 1.0, "b": 3.0, "c": 2.0}
    expected_result = {"a": 0.0, "b": 1.0, "c": 0.5, "X": 2.0}  # expected normalized dictionary
    assert dic_normalize(dic) == expected_result

    # Test that the function handles a dictionary with negative values
    dic = {"a": -1.0, "b": 2.0, "c": -3.0}
    expected_result = {'a': 0.4, 'b': 1.0, 'c': 0.0, 'X': -0.5}  # expected normalized dictionary
    assert dic_normalize(dic) == expected_result
    
def test_residue_features():
    # Define test inputs
    norm_res_prop = [
        {'A': 0.1, 'B': 0.2},
        {'A': 0.3, 'B': 0.4},
        {'A': 0.5, 'B': 0.6}
    ]
    residue = 'A'
    result = residue_features(norm_res_prop, residue)
    expected_output = np.array([1. , 0. , 0. , 0. , 0. , 0.1, 0.3, 0.5])
    np.testing.assert_array_equal(result, expected_output)
    
def test_one_of_k_encoding():
    atom = 'C'
    allowable_set = ['C', 'O', 'N']
    result = one_of_k_encoding(atom, allowable_set)

    expected_output = [True, False, False]

    assert result == expected_output

    with pytest.raises(ValueError):
        one_of_k_encoding('Z', allowable_set)
        
def test_atom_features():
    smile = "CC1=C2C=C(C=CC2=NN1)C3=CC(=CN=C3)OCC(CC4=CC=CC=C4)N"
    mol = Chem.MolFromSmiles(smile)  # type: ignore
    atom = next(mol.GetAtoms())  # get first atom from the smile
    result = atom_features(atom)
    expected_output = np.array(
        [True, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, # one-of-k encoding for symbol 'C'
         False, True, False, False, False, False, False, False, False, False, False, # one-of-k encoding for degree 1
         False, False, False,  True, False, False, False, False, False, False, False, # one-of-k encoding for total num Hs 3
         False, False, False,  True, False, False, False, False, False, False, False, # one-of-k encoding for implicit valence 3
         False])  # not aromatic

    np.testing.assert_array_equal(result, expected_output)
    

def test_smile_to_graph():
    smile = "CC1=C2C=C(C=CC2=NN1)C3=CC(=CN=C3)OCC(CC4=CC=CC=C4)N"
    result = smile_to_graph(smile)

    # Define the expected output
    expected_size = np.array(27)
    expected_features_shape = (27, 78)
    expected_edge_index_shape = (87, 2)
    # Check that the result matches the expected output
    assert result.size == expected_size
    assert result.features.shape == expected_features_shape
    assert result.edge_index.shape == expected_edge_index_shape
    
def test_pssm_calculation(data):
    result = pssm_calculation(data["aln_file"], data["target_sequence"])
    assert result.shape == (21, 961)
    
def test_seq_feature(data):
    norm_res_prop = [
        dic_normalize(re_prop) for re_prop in constants.residual_properties
    ]
    feature = seq_feature(norm_res_prop, data["target_sequence"])
    assert feature.shape == (961, 33)
    
def test_target_to_feature(data):
    pssm = pssm_calculation(data["aln_file"], data["target_sequence"])
    norm_res_prop = [
        dic_normalize(re_prop) for re_prop in constants.residual_properties
    ]
    other_feature = seq_feature(norm_res_prop, data["target_sequence"])
    feature = np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)
    assert feature.shape == (961, 54)
    
def test_target_to_graph(data):
    result = target_to_graph(data["data_path"], data["target_key"], data["target_sequence"])
    assert isinstance(result, Graph)
    assert result.size == np.array(961)
    assert result.edge_index.shape == (5421, 2)
    assert result.features.shape == (961, 54)

def test_valid_target(data):
    data_path = data["data_path"] 
    file_name = data["target_key"]
    exist = valid_target(file_name, data_path)
    not_exist = valid_target(file_name[1:], data_path)
    assert exist == True
    assert not_exist == False