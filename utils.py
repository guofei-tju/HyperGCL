import os
import torch as t
import pandas as pd
import cobra
import math
import random
from cobra.util.array import create_stoichiometric_matrix
from cobra.util.solver import linear_reaction_coefficients
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from process_data import get_coefficient_and_reactant, create_neg_rxn


def create_neg_rxns(args):
    print('reading dataset----------------------------------------------')
    with open(f'./data/{args.train}/{args.train}_rxn_name_list.txt', 'r') as f:
        pos_rxn = [i.strip().replace('\n', '') for i in f.readlines()]
    pos_index, pos_metas, pos_nums, rxn_directions = get_coefficient_and_reactant(pos_rxn)
    pos_metas_smiles = pd.read_csv(f'./data/{args.train}/{args.train}_meta_count.csv')
    chebi_meta_filter = pd.read_csv('./data/pool/cleaned_chebi.csv')
    name_to_smiles = pd.concat(
        [chebi_meta_filter.loc[:, ['name', 'smiles']], pos_metas_smiles.loc[:, ['name', 'smiles']]])

    print('creating negative rxns --------------------------------------')
    neg_rxn = create_neg_rxn(pos_rxn, pos_metas_smiles, chebi_meta_filter, args.balanced, args.negative_ratio,
                             args.atom_ratio)
    neg_index, neg_metas, neg_nums, rxn_directions = get_coefficient_and_reactant(neg_rxn)
    all_metas = list(set(sum(pos_metas, []) + sum(neg_metas, [])))
    all_metas.sort()

    pos_matrix = np.zeros((len(all_metas), len(pos_rxn)))
    rxn_df = pd.DataFrame(pos_matrix, index=all_metas, columns=['p_' + str(i) for i in range(len(pos_rxn))])
    for i in range(len(pos_index)):
        for j in range(len(pos_metas[i])):
            rxn_df.loc[pos_metas[i][j], 'p_' + str(i)] = float(pos_index[i][j])

    neg_matrix = np.zeros((len(all_metas), len(neg_rxn)))
    neg_df = pd.DataFrame(neg_matrix, index=all_metas, columns=['n_' + str(i) for i in range(len(neg_rxn))])
    for i in range(len(neg_index)):
        for j in range(len(neg_metas[i])):
            neg_df.loc[neg_metas[i][j], 'n_' + str(i)] = float(neg_index[i][j])
    label2rxn_df = pd.DataFrame(
        {'label': rxn_df.columns.to_list() + neg_df.columns.to_list(), 'rxn': pos_rxn + neg_rxn})

    return rxn_df, neg_df, name_to_smiles, label2rxn_df


def set_random_seed(seed=None):
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise ValueError('Seed must be a non-negative integer or omitted, not {}'.format(seed))
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    return seed


def get_filenames(path):
    return sorted(os.listdir(path))


def smiles_to_fp(smiles, radius=2, nBits=2048):
    # print(smiles)
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, useFeatures=False)
    fp_bits = fp.ToBitString()
    finger_print = np.array(list(map(int, fp_bits))).astype(np.float).reshape(1, -1)
    return finger_print


def remove_rxn(model, name_list):
    remove_list = []
    for i in range(len(model.metabolites)):
        meta = model.metabolites[i]
        if meta.name in name_list:
            continue
        remove_list.append(meta)
    model.remove_metabolites(remove_list, destructive=True)
    print(f'remove_rxn:{len(remove_list)}')


def get_data(path, sample):
    model = cobra.io.read_sbml_model(path + '/' + sample)
    # biomass = linear_reaction_coefficients(model)
    # model.remove_reactions(biomass, remove_orphans=True)
    # stoichiometric_matrix = create_stoichiometric_matrix(model)
    # incidence_matrix = np.abs(stoichiometric_matrix) > 0
    # remove_rxn_index = np.sum(incidence_matrix, axis=0) <= 1
    # model.remove_reactions(model.reactions[remove_rxn_index], remove_orphans=True)
    return model


def create_pool():
    print('-------------------------------------------------------')
    print('merging GEMs with reaction pool...')
    path = 'data/gems/xml-file'
    namelist = get_filenames(path)
    model_pool = cobra.io.read_sbml_model('./data/pool/universe.xml')
    pool_df = create_stoichiometric_matrix(model_pool, array_type='DataFrame')
    # yeast_gem = cobra.io.read_sbml_model('./data/gem/xml-file/yeast-GEM.xml')
    # yeast_gem_df = create_stoichiometric_matrix(yeast_gem, array_type='DataFrame')
    # combine = cobra.io.read_sbml_model('./results/bigg/comb_universe.xml')
    # combine_df = create_stoichiometric_matrix(combine, array_type='DataFrame')
    for sample in namelist:
        if sample.endswith('xml'):
            model = get_data(path, sample)
            model_pool.merge(model)
    cobra.io.write_sbml_model(model_pool, './results/bigg/comb_universe-fliter.xml')
    print('create pool done!')


def get_data_from_pool(path, sample, model_pool_df):
    if os.path.exists(path + '/reactions_w_gene_reaction_rule.csv'):
        rxns_df = pd.read_csv(path + '/reactions_w_gene_reaction_rule.csv')
        rxns = rxns_df.reaction[rxns_df.id == sample[:-4]].to_numpy()
    else:
        model = get_data(path, sample)
        rxns = np.array([rxn.id for rxn in model.reactions])
    model_df = model_pool_df[rxns]
    cols2use = model_pool_df.columns.difference(model_df.columns)
    return model_df, model_pool_df[cols2use]


def create_neg_incidence_matrix(incidence_matrix):
    incidence_matrix_neg = t.zeros(incidence_matrix.shape)
    for i, edge in enumerate(incidence_matrix.T):
        nodes = t.where(edge)[0]
        nodes_comp = t.tensor(list(set(range(len(incidence_matrix))) - set(nodes.tolist())))
        edge_neg_l = t.tensor(np.random.choice(nodes, math.floor(len(nodes) * 0.5), replace=False))
        edge_neg_r = t.tensor(
            np.random.choice(nodes_comp, len(nodes) - math.floor(len(nodes) * 0.5), replace=False))
        edge_neg = t.cat((edge_neg_l, edge_neg_r))
        incidence_matrix_neg[edge_neg, i] = 1
    return incidence_matrix_neg


def hyperlink_score_loss(y_pred, y):
    negative_score = t.mean(y_pred[y == 0])
    logistic_loss = t.log(1 + t.exp(negative_score - y_pred[y == 1]))
    loss = t.mean(logistic_loss)
    return loss


def create_label(incidence_matrix_pos, incidence_matrix_neg):
    y_pos = t.ones(len(incidence_matrix_pos.T))
    y_neg = t.zeros(len(incidence_matrix_neg.T))
    return t.cat((y_pos, y_neg))


def Tanimoto_smi(smiles_list):
    def _compute(data_1, data_2):
        norm_1 = (data_1 ** 2).sum(axis=1).reshape(data_1.shape[0], 1)
        norm_2 = (data_2 ** 2).sum(axis=1).reshape(data_2.shape[0], 1)
        prod = data_1.dot(data_2.T)
        return prod / (norm_1 + norm_2.T - prod)

    fps = [smiles_to_fp(s) for s in smiles_list]
    smi = np.ones((len(fps), len(fps)))
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            smi[i, j] = _compute(fps[i], fps[j])
            smi[j, i] = smi[i, j]
    return smi


def getGipKernel(y, trans, gamma):
    if trans:
        y = y.T
    krnl = t.mm(y, y.T)
    krnl = krnl / t.mean(t.diag(krnl))
    krnl = t.exp(-kernelToDistance(krnl) * gamma)
    return krnl


def kernelToDistance(k):
    di = t.diag(k).T
    d = di.repeat(len(k)).reshape(len(k), len(k)).T + di.repeat(len(k)).reshape(len(k), len(k)) - 2 * k
    return d


def cosine_kernel(tensor_1, tensor_2):
    return t.DoubleTensor([t.cosine_similarity(tensor_1[i], tensor_2, dim=-1).tolist() for i in
                           range(tensor_1.shape[0])])


def Mol_smiles(smiles_list):
    return np.array([smiles_to_fp(s)[0] for s in smiles_list])
