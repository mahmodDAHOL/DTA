from src.constants import *

test_data_path = project_path.joinpath("tests")

def test_data_exists():
    data_path = test_data_path.joinpath("data/davis")
    assert data_path.exists() == True

def test_aln_exists():
    data_path = test_data_path.joinpath("data/davis/aln")
    assert data_path.exists() == True
    assert len(list(data_path.glob("*.aln"))) > 0

def test_folds_exists():
    data_path = test_data_path.joinpath("data/davis/folds")
    assert data_path.exists() == True
    assert len(list(data_path.glob("*.txt"))) == 2

def test_affinities_exists():
    data_path = test_data_path.joinpath("data/davis/affinities")
    assert data_path.exists() == True

def test_ligands_exists():
    data_path = test_data_path.joinpath("data/davis/ligands.txt")
    assert data_path.exists() == True

def test_proteins_exists():
    data_path = test_data_path.joinpath("data/davis/proteins.txt")
    assert data_path.exists() == True