import json
import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np

# import pconsc4


project_path = Path(os.getcwd())
tool_path = project_path.parent.joinpath("tool")


def create_path(path: Path):
    if not path.exists():
        path.mkdir()


def seq_format(proteins_dic, output_dir: Path):
    for key, value in proteins_dic.items():
        with open(output_dir.joinpath(key).with_suffix('.fasta'), 'w') as f:
            f.writelines('>' + key + '\r\n')
            f.writelines(value + '\r\n')


def HHblitsMSA(bin_path: Path, db_path: Path, input_dir: Path, output_dir: Path):
    for fas_file in input_dir.glob("*"):
        process_file = input_dir.joinpath(fas_file)
        output_file = output_dir.joinpath(
            fas_file.with_suffix('.hhr'))
        output_file_a3m = output_dir.joinpath(fas_file.with_suffix('.a3m'))
        if output_file.exists() and output_file_a3m.exists():
            continue

        process_file = str(process_file).replace('(', '\(').replace(')', '\)')
        output_file = str(output_file).replace('(', '\(').replace(')', '\)')
        output_file_a3m = str(output_file_a3m).replace(
            '(', '\(').replace(')', '\)')
        cmd = str(bin_path) + ' -maxfilt 100000 -realign_max 100000 -d ' + str(db_path) + ' -all -B 100000 -Z 100000 -n 3 -e 0.001 -i ' + \
            str(process_file) + ' -o ' + str(output_file) + \
            ' -oa3m ' + str(output_file_a3m) + ' -cpu 8'
        print(cmd)
        os.system(cmd)


def HHfilter(bin_path: Path, input_dir: Path, output_dir: Path):
    file_prefix = []
    for file in input_dir.glob("*"):
        if file.suffix != 'a3m':
            continue
        temp_prefix = file.with_suffix('')
        if temp_prefix not in file_prefix:
            file_prefix.append(temp_prefix)

    for msa_file_prefix in file_prefix:
        file_name = msa_file_prefix.with_suffix('.a3m')
        process_file = input_dir.joinpath(file_name)
        output_file = output_dir.joinpath(file_name)
        if output_file.exists():
            continue
        process_file = str(process_file).replace('(', '\(').replace(')', '\)')
        output_file = str(output_file).replace('(', '\(').replace(')', '\)')
        cmd = str(bin_path) + ' -id 90 -i ' + \
            str(process_file) + ' -o ' + str(output_file)
        print(cmd)
        os.system(cmd)


def reformat(bin_path: Path, input_dir: Path, output_dir: Path):
    # print('reformat')
    for a3m_file in input_dir.glob("*"):
        process_file = input_dir.joinpath(a3m_file)
        output_file = output_dir.joinpath(a3m_file.with_suffix('.fas'))
        if output_file.exists():
            continue
        process_file = str(process_file).replace('(', '\(').replace(')', '\)')
        output_file = str(output_file).replace('(', '\(').replace(')', '\)')
        cmd = str(bin_path) + ' ' + str(process_file) + \
            ' ' + str(output_file) + ' -r'
        print(cmd)
        os.system(cmd)


def convertAlignment(bin_path: Path, input_dir: Path, output_dir: Path):
    # print('convertAlignment')
    for fas_file in input_dir.glob("*"):
        process_file = fas_file
        output_file = output_dir + '/' + fas_file.stem.with_suffix('.aln')
        if output_file.exists():
            continue
        process_file = str(process_file).replace('(', '\(').replace(')', '\)')
        output_file = str(output_file).replace('(', '\(').replace(')', '\)')
        cmd = 'python ' + str(bin_path) + ' ' + \
            str(process_file) + ' fasta ' + str(output_file)
        print(cmd)
        os.system(cmd)


def alnFilePrepare():
    print('aln file prepare ...')
    datasets = ['davis', 'kiba']
    for dataset in datasets:
        seq_dir = project_path.joinpath('data', dataset, 'seq')  # fasta files
        msa_dir = project_path.joinpath('data', dataset, 'msa')
        filter_dir = project_path.joinpath('data', dataset, 'hhfilter')
        reformat_dir = project_path.joinpath('data', dataset, 'reformat')
        aln_dir = project_path.joinpath('data', dataset, 'aln')
        protein_path = project_path.joinpath('data', dataset)
        proteins = json.load(open(protein_path.joinpath(
            'proteins.txt')), object_pairs_hook=OrderedDict)

        create_path(seq_dir)
        create_path(msa_dir)
        create_path(filter_dir)
        create_path(reformat_dir)
        create_path(aln_dir)

        HHblits_bin_path = tool_path.joinpath(
            'hhsuite/bin/hhblits')   # HHblits bin path
        # hhblits dataset for msa
        HHblits_db_path = tool_path.joinpath(
            'dataset/uniclust/uniclust30_2018_08/uniclust30_2018_08')
        HHfilter_bin_path = tool_path.joinpath(
            'hhsuite/bin/hhfilter')   # HHfilter bin path
        reformat_bin_path = tool_path.joinpath(
            'hhsuite/scripts/reformat.pl')   # reformat bin path
        convertAlignment_bin_path = project_path.joinpath(
            'tool/CCMpred/scripts/convert_alignment.py')   # ccmpred convertAlignment bin path

        def check_programs(*programs: list[Path]):
            for program in programs:
                if not program.exists():
                    raise Exception(
                        f'Program {str(program)} was not found. Please specify the run path.')

        check_programs(HHblits_bin_path, HHfilter_bin_path,
                       reformat_bin_path, convertAlignment_bin_path)

        seq_format(proteins, seq_dir)
        HHblitsMSA(HHblits_bin_path, HHblits_db_path, seq_dir, msa_dir)
        HHfilter(HHfilter_bin_path, msa_dir, filter_dir)
        reformat(reformat_bin_path, filter_dir, reformat_dir)
        convertAlignment(convertAlignment_bin_path, reformat_dir, aln_dir)

        print('aln file prepare over.')


def pconsc4Prediction():
    datasets = ['davis', 'kiba']
    model = pconsc4.get_pconsc4()
    for dataset in datasets:
        aln_dir = project_path.joinpath('data', dataset, 'hhfilter')
        output_dir = project_path.joinpath('data', dataset, 'pconsc4')
        if not output_dir.exists():
            output_dir.mkdir()
        file_list = list(aln_dir.glob("*"))
        random.shuffle(file_list)
        inputs = []
        outputs = []
        for file in file_list:
            input_file = aln_dir.joinpath(file)
            output_file = output_dir.joinpath(file.with_suffix('.npy'))
            if output_file.exists():
                # print(output_file, 'exist.')
                continue
            inputs.append(input_file)
            outputs.append(output_file)
            try:
                print('process', input_file)
                pred = pconsc4.predict(model, input_file)
                np.save(output_file, pred['cmap'])
                print(output_file, 'over.')
            except:
                print(output_file, 'error.')


if __name__ == '__main__':
    alnFilePrepare()
    pconsc4Prediction()
