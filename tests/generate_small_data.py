import json
import pickle
from collections import OrderedDict

import numpy as np

from src.constants import project_path
from src.data_process import *

sample_size = 600
pro_sample_size = 10
mol_sample_size = 60

data_path = project_path.joinpath("data/davis")
test_data_path = project_path.joinpath("tests/data/davis")
test_data_path.mkdir(exist_ok=True, parents=True)

proteins_file = data_path.joinpath("proteins.txt")
ligands_file = data_path.joinpath("ligands.txt")
affinities_file = data_path.joinpath("affinities")

train_arr = np.random.choice(range(500), size=(5, 100), replace=False)
train_list_of_lists = train_arr.tolist()

test_arr = np.random.choice(range(500, 600), size=(1, 100), replace=False)
test_list_of_lists = test_arr.tolist()

fold_path = test_data_path.joinpath('folds')
fold_path.mkdir(exist_ok=True)

with open(fold_path.joinpath('train_fold_setting1.txt'), 'w') as f:
    json.dump(train_list_of_lists, f)

with open(fold_path.joinpath('test_fold_setting1.txt'), 'w') as f:
    json.dump(test_list_of_lists, f)

    
proteins_dict = json.load(proteins_file.open(), object_pairs_hook=OrderedDict)
ligands_dict = json.load(ligands_file.open(), object_pairs_hook=OrderedDict)
affinity = pickle.load(affinities_file.open("rb"), encoding="latin1")

targets = {}
for i, (target_name, target_sequence) in enumerate(zip(proteins_dict.keys(), proteins_dict.values())):
    if i == pro_sample_size:
        break
    targets[target_name] = target_sequence

smiles = {}
for i, (smile_name, smile_sequence) in enumerate(zip(ligands_dict.keys(), ligands_dict.values())):
    if i == mol_sample_size:break
    smiles[smile_name] = smile_sequence


proteins_file_path = str(test_data_path.joinpath('proteins.txt'))
with open(proteins_file_path, 'w') as dst_file:
    json.dump(targets, dst_file, indent=0)
proteins = json.load(Path(proteins_file_path).open(), object_pairs_hook=OrderedDict)

ligands_file_path = str(test_data_path.joinpath('ligands.txt'))
with open(ligands_file_path, 'w') as dst_file:
    json.dump(smiles, dst_file, indent=0)
ligands = json.load(Path(ligands_file_path).open(), object_pairs_hook=OrderedDict)

    
src_file = data_path.joinpath('affinities')
dst_file = test_data_path.joinpath('affinities')
dst_file.parent.mkdir(exist_ok=True)
affinity = pickle.load(src_file.open("rb"), encoding="latin1")
sampled_affinity = np.random.choice(affinity.flatten(), sample_size, replace=False).reshape(60, 10)

with open(dst_file, 'wb') as file:
    pickle.dump(sampled_affinity, file)

aln_files = [data_path.joinpath(f"aln/{pro}.aln") for pro in targets]
contact_files = [data_path.joinpath(f"pconsc4/{pro}.npy") for pro in targets]
for src_path in contact_files:
    dst_path = test_data_path.joinpath('pconsc4')
    dst_path.mkdir(exist_ok=True)

    with open(src_path, 'rb') as src_file:
        file_contents = src_file.read()

    with open(dst_path / src_path.name, 'wb') as dst_file:
        dst_file.write(file_contents)

for src_path in aln_files:
    dst_path = test_data_path.joinpath('aln')
    dst_path.mkdir(exist_ok=True)

    with open(src_path, 'rb') as src_file:
        file_contents = src_file.read()

    with open(dst_path / src_path.name, 'wb') as dst_file:
        dst_file.write(file_contents)