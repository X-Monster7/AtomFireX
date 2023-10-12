#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/10/11 13:28
@version: 1.0.0
"""
import re
from typing import List

from rdkit import Chem
from tqdm import tqdm
import pandas as pd


# 用于打开SDF文件的函数，每次读取一行
def read_sdf_by_chunks(file_path, mol_size):
    mol_supplier = Chem.SDMolSupplier(file_path)
    mols: List[Chem.rdchem.Mol] = []
    for mol in mol_supplier:
        if mol is not None:
            mols.append(mol)
            if len(mols) >= mol_size:
                yield mols
                mols = []
    # 最后一次的mol小于mol_size,但也要返回
    if mols:
        is_all_mol_processed = True
        yield mols


if __name__ == '__main__':
    sdf_file = "H:\MSN-Data\MoNA-export-All_Spectra.sdf"
    # sdf_file = "H:\MSN-Data\MoNA-export-HMDB.sdf"
    count_num_of_mols = 0
    MAX_CHUNK_MOL_SIZE = 1024
    MAX_SAVE_MOL_SIZE = 1024
    is_all_mol_processed = False
    # 创建一个空的Pandas DataFrame来存储数据
    data = pd.DataFrame()

    # 逐行读取SDF文件并处理数据
    for mols in read_sdf_by_chunks(sdf_file, mol_size=MAX_CHUNK_MOL_SIZE):
        for mol in tqdm(mols):
            if mol is not None:
                # 获取分子的所有属性作为字典
                properties = mol.GetPropsAsDict()
                # 将properties comment中的SMILE 和 computed SMILE提取出来，形成字典，并补充到data中
                comment = properties.get('COMMENT')
                # 匹配独立的 "SMILES=" 行，只允许前面有空格
                properties['SMILES'] = re.findall(r"^\s*SMILES=(.*?)\n", comment, re.DOTALL)
                properties['COMPUTED_SMILES'] = re.findall(r"^\s*computed SMILES=(.*?)\n", comment, re.DOTALL)
                data: pd.DataFrame = data.append(properties, ignore_index=True)
        count_num_of_mols += len(mols)
        print(f'total {count_num_of_mols} mols')
        if count_num_of_mols % MAX_SAVE_MOL_SIZE == 0 or is_all_mol_processed:
            data.to_csv(f'./ms_csv_data/{count_num_of_mols}.csv', index=False)
            print(f"ms_data since last save time has saved to ./ms_csv_data/{count_num_of_mols}.csv")
            data = pd.DataFrame()
