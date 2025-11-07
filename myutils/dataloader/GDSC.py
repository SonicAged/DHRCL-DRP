import os
import csv
import pickle
from typing import Tuple

import pandas as pd

class DataLoader:
    """Load GDSC dataset
    
    Args:
        Drug_info_file: Drug information file path
        Drug_feature_file: Drug feature file path
        Cell_line_info_file: Cell line information file path
        Genomic_mutation_file: Genomic mutation file path
        Gene_expression_file: Gene expression file path
        Methylation_file: Methylation file path
        Cancer_response_exp_file: Cancer response file path
        IC50_threds_file: IC50 threshold file path
    
    Returns:
        Tuple of drug features, mutation features, 
        gene expression features, methylation features, 
        data, number of cell lines, 
        number of drugs
    """
    def __init__(self, 
                 Drug_info_file: str, 
                 Drug_feature_file: str, 
                 Cell_line_info_file: str, 
                 Genomic_mutation_file: str,
                 Gene_expression_file: str, 
                 Methylation_file: str, 
                 Cancer_response_exp_file: str, 
                 IC50_threds_file: str):
        self.Drug_info_file = Drug_info_file
        self.Drug_feature_file = Drug_feature_file
        self.Cell_line_info_file = Cell_line_info_file
        self.Genomic_mutation_file = Genomic_mutation_file
        self.Gene_expression_file = Gene_expression_file
        self.Methylation_file = Methylation_file
        self.Cancer_response_exp_file = Cancer_response_exp_file
        self.IC50_threds_file = IC50_threds_file

    def _load_drug_feature(self, drug_feature_file: str) -> Tuple[dict, list]:
        """Load drug features from pickle files."""
        drug_pubchem_id_set = []
        drug_feature = {}
        for each in os.listdir(drug_feature_file):
            if not each.endswith(('.pkl', '.pickle')):
                continue
                
            drug_id = each.split('.')[0]
            drug_pubchem_id_set.append(drug_id)
            
            try:
                pkl_path = os.path.join(drug_feature_file, each)
                with open(pkl_path, 'rb') as pkl_file:
                    feat_mat, adj_list, degree_list, func = pickle.load(pkl_file)
                drug_feature[drug_id] = [feat_mat, adj_list, degree_list]         
            except Exception as e:
                print(f"Error reading drug feature file {each}: {e}")
                continue
        
        if len(drug_pubchem_id_set) == 0:
            raise ValueError("Failed to load any drug feature files")
            
        assert len(drug_pubchem_id_set) == len(drug_feature.values()), \
            "Drug ID set and feature count do not match"
        
        return drug_feature, drug_pubchem_id_set

    def _load_cell_line_feature(self, 
                                mutation_feature_file: str, 
                                gexpr_feature_file: str, 
                                methylation_feature_file: str
                                ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load cell line omics features."""
        mutation_feature = pd.read_csv(mutation_feature_file, 
                                    sep=',', header=0, index_col=[0]
                                    )
        gexpr_feature = pd.read_csv(gexpr_feature_file, 
                                    sep=',', header=0, index_col=[0]
                                    )
        methylation_feature = pd.read_csv(methylation_feature_file, 
                                        sep=',', header=0, index_col=[0]
                                        )
        mutation_feature = mutation_feature.loc[list(gexpr_feature.index)]
        assert methylation_feature.shape[0] == gexpr_feature.shape[0] == mutation_feature.shape[0]
        return mutation_feature, gexpr_feature, methylation_feature

    def _load_response_data(self, 
                            drug_info_file: str,
                            ic50_threds_file: str,
                            cell_line_info_file: str,
                            cancer_response_exp_file: str,
                              drug_pubchem_id_set: list, 
                              mutation_feature: pd.DataFrame
                              ) -> list:
        """Load and process cancer response data."""
        # Load drug info to map drug ID to PubChem ID
        try:
            with open(drug_info_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = [item for item in reader]
        except UnicodeDecodeError:
            with open(drug_info_file, 'r', encoding='latin-1') as f:
                reader = csv.reader(f)
                rows = [item for item in reader]
        drugid2pubchemid = {item[0]: item[5] for item in rows if item[5].isdigit()}

        # Load drug IC50 thresholds
        drug2thred = {}
        try:
            with open(ic50_threds_file, 'r', encoding='utf-8') as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        drug2thred[str(parts[0])] = float(parts[1])
        except Exception as e:
            print(f"Error reading IC50 threshold file: {e}")
            raise

        # Load cell line cancer type
        cellline2cancertype = {}
        for line in open(cell_line_info_file).readlines()[1:]:
            cellline_id = line.split('\t')[1]
            TCGA_label = line.strip().split('\t')[-1]
            cellline2cancertype[cellline_id] = TCGA_label

        # Load and process response data
        response_data = pd.read_csv(cancer_response_exp_file, 
                                    sep=',', header=0, index_col=[0]
                                    )
        drug_match_list = [item for item in response_data.index 
                           if item.split(':')[1] in drugid2pubchemid.keys()]
        response_data = response_data.loc[drug_match_list]
        
        data_idx = []
        use_thred = True
        
        for drug in response_data.index:
            for cellline in response_data.columns:
                pubchem_id = drugid2pubchemid[drug.split(':')[-1]]
                
                if str(pubchem_id) in drug_pubchem_id_set and cellline in mutation_feature.index:
                    if cellline in cellline2cancertype.keys():
                        try:
                            ln_IC50 = float(response_data.loc[drug, cellline])
                            if use_thred:
                                threshold = drug2thred.get(pubchem_id, -2.0)
                                binary_IC50 = 1 if ln_IC50 < threshold else -1

                            data_idx.append(
                                (cellline, pubchem_id, 
                                 binary_IC50, cellline2cancertype[cellline])
                                 )
                        except (ValueError, TypeError):
                            continue

        # Eliminate ambiguity responses
        data_sort = sorted(
            data_idx, 
            key=(lambda x: [x[0], x[1], x[2]]), 
            reverse=True
            )
        data_tmp = []
        data_new = []
        data_idx1 = [[i[0], i[1]] for i in data_sort]
        for i, k in zip(data_idx1, data_sort):
            if i not in data_tmp:
                data_tmp.append(i)
                data_new.append(k)
        return data_new

    def __call__(self) -> Tuple:
        print('Loading data...')

        print('1-Loading drug information')
        drug_feature, drug_pubchem_id_set = self._load_drug_feature(self.Drug_feature_file)

        print('2-Loading cell lines information')
        mutation_feature, gexpr_feature, methylation_feature = \
            self._load_cell_line_feature(self.Genomic_mutation_file, 
                                         self.Gene_expression_file, 
                                         self.Methylation_file
                                         )

        print('3-Loading response information')
        response = self._load_response_data(self.Drug_info_file, 
                                            self.IC50_threds_file, 
                                            self.Cell_line_info_file, 
                                            self.Cancer_response_exp_file, 
                                            drug_pubchem_id_set, 
                                            mutation_feature
                                            )

        num_celllines = len(set([item[0] for item in response]))
        num_drugs = len(set([item[1] for item in response]))
        print('All %d pairs across %d cell lines and %d drugs.'
            % (len(response), num_celllines, num_drugs)
            )

        return drug_feature, mutation_feature, \
                gexpr_feature, methylation_feature, \
                response, num_celllines, num_drugs