#!/usr/bin/config python
# -*- coding: utf-8 -*-


"""
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/10/11 13:28
@version: 1.0.0
"""
import re
import os
import traceback
from typing import List, Dict, Any
from pprint import pprint
from rdkit import Chem
from tqdm import tqdm
import pandas as pd
from util.logger_util.index import Log


# 用于打开SDF文件的函数，每次读取一行
def read_sdf_by_chunks(file_path, mol_size):
    """

    Args:
        file_path:
        mol_size:

    Returns:

    """
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
        global is_all_mol_processed
        is_all_mol_processed = True
        yield mols


def exact_property(feature: List[str], properties: dict) -> dict:
    exact_feature = dict()

    comment = properties.get('COMMENT')
    # 匹配独立的 "SMILES=" 行，只允许前面有空格
    _SMILE = re.findall(r"^\s*SMILES=(.*?)\n", comment, re.DOTALL)
    properties['SMILES'] = _SMILE[0] if _SMILE else " "
    _COMPUTED_SMILES = re.findall(r"^\s*computed SMILES=(.*?)\n", comment, re.DOTALL)
    properties['COMPUTED_SMILES'] = _COMPUTED_SMILES[0] if _COMPUTED_SMILES else " "
    # 按照feature中描述的需要提取的特征，将properties中的特征提取到data中
    # 不使用链式drop去除properties中的属性这种做法，不优雅而且不能面对变更
    for _feature in feature[1:]:
        if _feature in properties:
            exact_feature[_feature] = properties[_feature]
    exact_feature['ID'] = ID
    # print(exact_feature)
    return exact_feature


if __name__ == '__main__':

    count_num_of_mols = 0
    MAX_CHUNK_MOL_SIZE = 2048 / 512
    MAX_CSV_SAVE_MOL_SIZE = 20480 / 512
    is_all_mol_processed = False

    sdf_file = "H:\\MSN-Data\\MoNA.sdf"
    # 每MAX_SAVE_MOL_SIZE个txt文件换一次存储位置，使其和csv文件的分子是对应的，方便合并特征
    ms_txt_data_dir = f'./ms_txt_data/{0 + MAX_CSV_SAVE_MOL_SIZE}'
    ms_csv_data_dir = './ms_csv_data'
    # Create directories for saving data if they don't exist
    os.makedirs(ms_txt_data_dir, exist_ok = True)
    os.makedirs(ms_csv_data_dir, exist_ok = True)

    feature = ["ID", "FORMULA", "MASS SPECTRAL PEAKS", "SMILES",
               "EXACT MASS", "ION MODE", "COMPUTED_SMILES"]
    ID = 0

    logger = Log("./log/error_data.log", "DEBUG").get_logger()

    # 创建一个空的Pandas DataFrame来存储数据
    data = pd.DataFrame(columns = feature)
    # 逐块读取SDF文件并处理数据
    for mols in read_sdf_by_chunks(sdf_file, mol_size = MAX_CHUNK_MOL_SIZE):
        # MAX_CHUNK_MOL_SIZE个分子
        for mol in tqdm(mols):
            properties = mol.GetPropsAsDict()
            exacted_properties = exact_property(feature, properties)
            data.loc[len(data)] = exacted_properties
            try:
                with open(
                        f'{ms_txt_data_dir}/{ID}_m{properties["EXACT MASS"]}'
                        f'_f{properties["FORMULA"]}', 'w'
                ) as file:
                    file.write(properties["MASS SPECTRAL PEAKS"])
                ID += 1
            except BaseException as e:
                # pprint(f"出错的分子特征：{properties}")
                # pprint(f"错误路径: {traceback.print_exc()}")
                # pprint(f"错误参数：{e.args}")
                logger.error(exacted_properties)

        count_num_of_mols += len(mols)
        print(f'total {count_num_of_mols} mols')
        if count_num_of_mols % MAX_CSV_SAVE_MOL_SIZE == 0 or is_all_mol_processed:
            # 每MAX_SAVE_MOL_SIZE改变一次txt文件夹的位置
            ms_txt_data_dir = f'./ms_txt_data/{count_num_of_mols + MAX_CSV_SAVE_MOL_SIZE}'
            os.makedirs(ms_txt_data_dir, exist_ok = True)
            # 每MAX_SAVE_MOL_SIZE记录一份csv文件
            data.to_csv(f'{ms_csv_data_dir}/{count_num_of_mols}.csv', index = False)
            print(f"ms_data since last save time has saved to {ms_csv_data_dir}/{count_num_of_mols}.csv")
            data = pd.DataFrame(columns = feature)
            # print(data)
