import random
import os
import json
import torch
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold

from save_data import load_dataset
from submodel.hypergraph.drug import Encoder as DrugEncoder
from submodel.hypergraph.cellline import Encoder as CelllineEncoder
from submodel.hypergraph.response import Encoder as ResponseEncoder
from model import MyModel
from sampler import Sampler
from optimizer import Optimizer

def _set_torch_config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def _get_params(config_path):
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    arg_train = config['optimizer']
    arg_model = config['model']
    arg_data = config['data']
    return arg_train, arg_model, arg_data

def _set_train_params(arg_train):
    device = torch.device(arg_train['device'] \
                      if torch.cuda.is_available() else "cpu"
                      )
    print('train device:' + str(device))

    lr = arg_train['lr']
    seed = arg_train['seed']
    save_model = arg_train['save_model']
    save_path = arg_train['save_path']
    save_tensor = arg_train['save_tensor']
    k_folds = arg_train['k_folds']
    n_repeats = arg_train['n_repeats']
    lambda_drug = arg_train['lambda_drug']
    lambda_cellline = arg_train['lambda_cellline']
    return device, lr, seed, \
        save_model, save_path, \
        save_tensor, k_folds, n_repeats, \
        lambda_drug, lambda_cellline


def _set_model_params(arg_model):
    hidden_channels = arg_model['hidden_channels']
    output_channels = arg_model['output_channels']
    num_scales = arg_model['num_scales']
    k = arg_model['k']
    beta = arg_model['beta']
    theta = arg_model['theta']
    omics = arg_model['omics']
    print('model parameters:')
    print(json.dumps(arg_model, indent=4))
    return hidden_channels, output_channels, num_scales, k, beta, theta, omics

def _get_shape(drug_set, cellline_set):
    gexpr_shape = cellline_set.dataset.tensors[1].shape[-1]
    methylation_shape = cellline_set.dataset.tensors[2].shape[-1]
    atom_shape = drug_set.dataset[0]['x'].shape[-1]
    return gexpr_shape, methylation_shape, atom_shape

def _get_num(drug_set, cellline_set):
    dataset_size = 0
    for _ in drug_set.dataset:
        dataset_size += 1
    num_drugs = dataset_size
    num_celllines = cellline_set.batch_size
    return num_drugs, num_celllines

def _set_path(save_path, save_model, arg_data):
    save_path = save_path if save_model else None
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
        for file in arg_data:
            os.makedirs(save_path + file['name'] + '/model/')
            os.makedirs(save_path + file['name'] + '/loss/')
            os.makedirs(save_path + file['name'] + '/auc/')
            os.makedirs(save_path + file['name'] + '/result/')
    return save_path
    
def _save_result(res, output_path, args_list):
    res_mean = np.array([r[1:] for r in res]).mean(axis=0)
    res_mean = ['mean'] + list(res_mean)
    res.append(res_mean)

    res_std = np.array([r[1:] for r in res]).std(axis=0)
    res_std = ['std'] + list(res_std)
    res.append(res_std)

    res_final = ['final'] \
        + [f"{mean:.4f}±{std:.4f}" \
            for mean, std in zip(res_mean[1:], res_std[1:])]
    res.append(res_final)

    df = pd.DataFrame(data=[args_list])
    df.to_csv(output_path, header=False, index=False, mode='w')
    df = pd.DataFrame(data=res)
    df.to_csv(output_path, header=['fold', 'auc', 'aupr', 'f1', \
                                'acc', 'prec', 'recall', 'mcc'], 
                                encoding="utf-8-sig", 
                                index=False, mode='a'
                )
    
    print(f"Cross-validation completed, results saved to: {output_path}")
    print(f'Average results: '
        f'AUC={res_mean[1]:.4f}, '
        f'AUPR={res_mean[2]:.4f}, '
        f'F1={res_mean[3]:.4f}, '
        f'ACC={res_mean[4]:.4f}, '
        f'MCC={res_mean[5]:.4f}, '
        f'PRE={res_mean[6]:.4f}, '
        f'REC={res_mean[7]:.4f}'
        )

def _save_loss(loss_list, output_path):
    # For the new dictionary-based loss structure
    epochs = range(len(loss_list['total_loss']))
    loss_df = pd.DataFrame({'epoch': epochs})
    
    # Add each loss component as a separate column
    for loss_name, loss_values in loss_list.items():
        loss_df[loss_name] = loss_values
    
    loss_df.to_csv(output_path, index=False)

def _save_auc(auc_list, output_path):
    # For the AUC history structure
    auc_df = pd.DataFrame(auc_list)
    auc_df.to_csv(output_path, index=False)

def _insert_repeat(final_metrics, best_epoch):
    res_repeat = [f'R{repeat+1}F{fold+1}', 
                    final_metrics['auc'], 
                    final_metrics['aupr'], 
                    final_metrics['f1'], 
                    final_metrics['acc'], 
                    final_metrics['mcc'], 
                    final_metrics['precision'], 
                    final_metrics['recall']
                    ]
    print(
        f'Results: \n'
        f'\tAUC: {final_metrics["auc"]:.4f}, '
        f'AUPR: {final_metrics["aupr"]:.4f}, '
        f'F1: {final_metrics["f1"]:.4f}\n'
        f'\tACC: {final_metrics["acc"]:.4f}, '
        f'MCC: {final_metrics["mcc"]:.4f}, '
        f'PRE: {final_metrics["precision"]:.4f}\n'
        f'\tREC: {final_metrics["recall"]:.4f}'
    )
    print(f'Best round: {best_epoch}\n')
    return res_repeat

if __name__ == '__main__':
    config_path = 'config.yaml'
    arg_train, arg_model, arg_data \
        = _get_params(config_path)
    device, lr, seed, \
    save_model, save_path, \
    save_tensor, \
    k_folds, n_repeats, lambda_drug, lambda_cellline \
        = _set_train_params(arg_train)
    hidden_channels, output_channels, num_scales, k, beta, theta, omics \
        = _set_model_params(arg_model)

    _set_torch_config(seed)
    save_path = _set_path(save_path, save_model, arg_data)

    for files in arg_data:
        print("Current dataset: " + files['name'])
        drug_set, cellline_set, allpairs = load_dataset(files)
        epochs = files['epochs']
        num_drugs, num_celllines \
            = _get_num(drug_set, cellline_set)
        gexpr_shape, methylation_shape, atom_shape \
            = _get_shape(drug_set, cellline_set)

        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        res = []
        args_list = 'ep' + str(epochs) + '_' \
                    + 'lr' + str(lr) + '_' \
                    + 'cv'
        print(args_list)

        fold_count = 0
        for repeat in range(n_repeats):
            print(f'Repeat {repeat + 1}/{n_repeats}')
            
            # set the random seed for each repeat
            kfold_repeat = KFold(n_splits=k_folds, 
                                shuffle=True, random_state=42 + repeat)
            
            for fold, (train_indices, test_indices) in \
                enumerate(kfold_repeat.split(np.arange(len(allpairs)))):
                fold_count += 1
                print(f'Round {repeat + 1}, fold {fold + 1}'
                      f' (Total {fold_count} times)'
                      )
                
                # create cross-validation sampler
                sampler = Sampler(allpairs=allpairs, 
                                  num_celllines=num_celllines, 
                                  num_drugs=num_drugs, 
                                  train_indices=train_indices, 
                                  test_indices=test_indices
                                )
                
                # create model
                model \
                = MyModel(hidden_channels=hidden_channels,
                          output_channels=output_channels,
                            cellline_encoder=\
                                CelllineEncoder(dim_gexp=gexpr_shape, 
                                                dim_methy=methylation_shape, 
                                                output=output_channels, 
                                                num_scales=num_scales,
                                                k=k,
                                                omics=omics
                                                ).to(device),
                            drug_encoder=\
                                DrugEncoder(input_channel=atom_shape, 
                                            output_channel=output_channels,
                                            units_list=[128, 128]
                                            ).to(device),
                            response_encoder=\
                                ResponseEncoder(output_channels, 
                                                output_channels,
                                                sampler.r, beta, theta
                                                ).to(device),
                            lambda_drug=lambda_drug,
                            lambda_cellline=lambda_cellline
                        ).to(device)
                
                # create optimizer
                model_save_path = save_path + files['name'] \
                    + f'/model/' \
                    + f'R{repeat+1}F{fold+1}'
                opt = Optimizer(model=model, 
                                drug_set=drug_set, 
                                cellline_set=cellline_set, 
                                sampler=sampler, 
                                lr=lr, 
                                epochs=epochs, 
                                save_model=save_model, 
                                save_path=model_save_path, 
                                device=device,
                                save_tensor=save_tensor
                                ).to(device)
                
                # execute training
                best_epoch, test_true, test_pred, final_metrics, loss_list, auc_list \
                    = opt(progress_desc=f'Training progress R{repeat+1}F{fold+1}')

                res.append(_insert_repeat(final_metrics, best_epoch))

                loss_csv_path = save_path + files['name'] \
                    + f'/loss/loss_history_R{repeat+1}F{fold+1}.csv'
                _save_loss(loss_list, loss_csv_path)
                
                auc_csv_path = save_path + files['name'] \
                    + f'/auc/auc_history_R{repeat+1}F{fold+1}.csv'
                _save_auc(auc_list, auc_csv_path)

        output_path = save_path + files['name'] \
            + f'/result/{args_list}.csv'
        _save_result(res, output_path, args_list)

