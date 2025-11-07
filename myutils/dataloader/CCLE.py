from typing import Tuple

import pandas as pd
import deepchem as dc
from rdkit import Chem

class DataLoader:
    """
    Load CCLE dataset
    
    Args:
        Drug_feature_file: Drug feature file path
        Genomic_mutation_file: Genomic mutation file path
        Gene_expression_file: Gene expression file path
        Methylation_file: Methylation file path
        Cancer_response_exp_file: Cancer response file path
    
    Returns:
        Tuple of drug features, mutation features, 
        gene expression features, methylation features, 
        data, number of cell lines, 
        number of drugs
    """
    def __init__(self, 
                 Drug_feature_file: str, 
                 Genomic_mutation_file: str, 
                 Gene_expression_file: str, 
                 Methylation_file: str, 
                 Cancer_response_exp_file: str):
        self.Drug_feature_file = Drug_feature_file
        self.Genomic_mutation_file = Genomic_mutation_file
        self.Gene_expression_file = Gene_expression_file
        self.Methylation_file = Methylation_file
        self.Cancer_response_exp_file = Cancer_response_exp_file
        
    def _load_drug_feature(self, drug_feature_file: str) -> dict:
        drug = pd.read_csv(drug_feature_file, sep=',', header=0)
        drug_feature = {}
        featurizer = dc.feat.ConvMolFeaturizer()
        for tup in zip(drug['pubchem'], drug['isosmiles']):
            mol = Chem.MolFromSmiles(tup[1])
            X = featurizer.featurize(mol)
            drug_feature[str(tup[0])] = [X[0].get_atom_features(), 
                                        X[0].get_adjacency_list(), 1
                                        ]
        return drug_feature

    def _load_cell_line_feature(self, mutation_feature_file: str, 
                                gexpr_feature_file: str, 
                                methylation_feature_file: str
                                ) -> Tuple:
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
        return mutation_feature, gexpr_feature, methylation_feature

    def _load_response_data(self, cancer_response_exp_file: str) -> Tuple:
        response_z_score = pd.read_csv(cancer_response_exp_file, 
                                       sep=',', header=0
                                       )
        response_idx = []
        thred = 0.8
        for tup in zip(response_z_score['DepMap_ID'], 
                       response_z_score['pubchem'], 
                       response_z_score['Z_SCORE']
                       ):
            t = 1 if tup[2] > thred else -1
            response_idx.append((tup[0], str(tup[1]), t))

        # eliminate ambiguity responses
        response_sort = sorted(response_idx, 
                               key=(lambda x: [x[0], x[1], x[2]]), 
                               reverse=True
                               )
        response_exist = []
        response = []
        response_idx1 = [[i[0], i[1]] for i in response_sort]
        for i, k in zip(response_idx1, response_sort):
            if i not in response_exist:
                response_exist.append(i)
                response.append(k)
        return response
    
    def __call__(self) -> Tuple:
        print('Loading data...')

        print('1-Loading drugs information')
        drug_feature = self._load_drug_feature(self.Drug_feature_file)

        print('2-Loading cell lines information')
        mutation_feature, gexpr_feature, methylation_feature = \
            self._load_cell_line_feature(self.Genomic_mutation_file, 
                                         self.Gene_expression_file, 
                                         self.Methylation_file
                                         )

        print('3-Loading response information')
        response = self._load_response_data(self.Cancer_response_exp_file)

        num_celllines = len(set([item[0] for item in response]))
        num_drugs = len(set([item[1] for item in response]))
        print('All %d pairs across %d cell lines and %d drugs.'
            %(len(response), num_celllines, num_drugs)
            )

        return drug_feature, mutation_feature, \
                gexpr_feature, methylation_feature, \
                response, num_celllines, num_drugs