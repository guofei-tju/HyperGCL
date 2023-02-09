import os
import torch
import torch.nn as nn
import config
import pandas as pd
import numpy as np
import copy
from CLOSEgaps import CLOSEgaps
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, average_precision_score

from utils import set_random_seed, create_neg_rxns, smiles_to_fp, getGipKernel


def test_pre(feature, incidence_matrix, model):
    model.eval()
    with torch.no_grad():
        y_pred = model.predict(feature, incidence_matrix)
    return torch.squeeze(y_pred)


def train(args, X_smiles, train_incidence_pos, incidence_train, incidence_valid, y_train, y_valid):
    print('---------------- calculating similarity ------------------------------')
    X_smiles_t = torch.tensor([smiles_to_fp(s)[0] for s in X_smiles], dtype=torch.float)
    X_similarity_t = getGipKernel(X_smiles_t, False, args.g_lambda)

    node_num, hyper_num = incidence_train.shape
    model = CLOSEgaps(input_num=node_num, input_feature_num=train_incidence_pos.shape[1],
                     similarity=X_similarity_t, emb_dim=args.emb_dim, conv_dim=args.conv_dim,
                     head=args.head, p=args.p, L=args.L,
                     use_attention=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crossentropyloss = nn.CrossEntropyLoss()
    max_valid_f1 = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    print('---------------- start training ------------------------------')
    for _ in tqdm(range(args.epoch)):
        model.train()
        epoch_size = incidence_train.shape[1] // args.batch_size
        for e in range(epoch_size):
            optimizer.zero_grad()
            y_pred = model(train_incidence_pos, incidence_train[:, e * args.batch_size:(e + 1) * args.batch_size])
            loss = crossentropyloss(y_pred, y_train[e * args.batch_size:(e + 1) * args.batch_size])
            loss.backward()
            optimizer.step()
        valid_score = test_pre(train_incidence_pos, incidence_valid, model)
        true_valid_score = valid_score.numpy()[:, 1]
        b_score = [int(s >= 0.5) for s in true_valid_score]
        auc_score = roc_auc_score(y_valid, true_valid_score)
        pr = precision_score(y_valid, b_score)
        re = recall_score(y_valid, b_score)
        f1 = f1_score(y_valid, b_score)
        aupr = average_precision_score(y_valid, true_valid_score)
        if max_valid_f1 < f1:
            max_valid_f1 = f1
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f'\nvalid, epoch:{_}, f1:{f1},pr:{pr},recall:{re},auc:{auc_score},aupr:{aupr}')
    model.load_state_dict(best_model_wts)
    return model


def predict(model, train_incidence_pos, incidence_test, y_test):
    print('---------------- start testing ------------------------------')
    y_pred = test_pre(train_incidence_pos, incidence_test, model)
    score_t = torch.squeeze(y_pred)
    true_test_score = score_t.numpy()[:, 1]
    b_score = [int(s >= 0.5) for s in true_test_score]
    auc_score = roc_auc_score(y_test, true_test_score)
    pr = precision_score(y_test, b_score)
    re = recall_score(y_test, b_score)
    f1 = f1_score(y_test, b_score)
    aupr = average_precision_score(y_test, true_test_score)
    print(
        f'f1:{f1},pr:{pr},recall:{re},auc:{auc_score},aupr:{aupr}')
    return f1, pr, re, auc_score, aupr


if __name__ == '__main__':
    args = config.parse()
    set_random_seed(args.seed)
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    for i in range(args.iteration):
        print(f'---------------- iteration {i} ------------------------------')
        if args.create_negative:
            print('---------------- creating negative samples ------------------------------')
            rxn_df, neg_df, name_to_smiles, label2rxn_df = create_neg_rxns(args)
            rxn_df[rxn_df != 0] = 1
            neg_df[neg_df != 0] = 1
            train_pos_df, valid_pos_df, test_pos_df = np.split(rxn_df.sample(frac=1, axis=1, random_state=args.seed),
                                                               [int(.6 * len(rxn_df.T)),
                                                                int(.8 * len(rxn_df.T))], axis=1)
            train_neg_df, valid_neg_df, test_neg_df = np.split(neg_df.sample(frac=1, axis=1, random_state=args.seed),
                                                               [int(.6 * len(neg_df.T)),
                                                                int(.8 * len(neg_df.T))], axis=1)
            train_df = pd.concat([train_pos_df, train_neg_df], axis=1).sample(frac=1, axis=1)
            test_df = pd.concat([test_pos_df, test_neg_df], axis=1).sample(frac=1, axis=1)
            valid_df = pd.concat([valid_pos_df, valid_neg_df], axis=1).sample(frac=1, axis=1)
        else:
            train_df = pd.read_csv(f'./data/{args.train}/train_df_ub_G.csv', index_col=0)
            test_df = pd.read_csv(f'./data/{args.train}/test_df_ub_G.csv', index_col=0)
            valid_df = pd.read_csv(f'./data/{args.train}/valid_df_ub_G.csv', index_col=0)
            train_pos_df = train_df.loc[:, lambda d: d.columns.str.contains('p')]
            name_to_smiles = pd.read_csv(f'./data/{args.train}/name2smiles_ub_G.csv')

        y_train = torch.tensor(['p' in c for c in train_df.columns], dtype=torch.long).view(-1)
        y_test = torch.tensor(['p' in c for c in test_df.columns], dtype=torch.float).view(-1)
        y_valid = torch.tensor(['p' in c for c in valid_df.columns], dtype=torch.float).view(-1)

        train_incidence_pos = torch.tensor(train_pos_df.to_numpy(), dtype=torch.float)
        incidence_train = torch.tensor(train_df.to_numpy(), dtype=torch.float)
        incidence_test = torch.tensor(test_df.to_numpy(), dtype=torch.float)
        incidence_valid = torch.tensor(valid_df.to_numpy(), dtype=torch.float)
        X_smiles = [name_to_smiles[name_to_smiles['name'] == name].smiles.values[0] for name in train_df.index]

        model = train(args, X_smiles, train_incidence_pos, incidence_train, incidence_valid, y_train, y_valid)
        f1, pr, re, auc_score, aupr = predict(model, train_incidence_pos, incidence_test, y_test)
        torch.save({'model': model.state_dict()}, f'{args.output}{args.train}_model_{i}.pth')
