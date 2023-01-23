"""Evaluate DFR on spurious correlations datasets."""

import torch

import numpy as np
import os
import tqdm
import argparse
import pickle
import json

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils import evaluate_no_min_group

from poison_datasets import get_train_dataset, get_test_dataset
from models import *

# WaterBirds
C_OPTIONS = [1., 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]
CLASS_WEIGHT_OPTIONS = [1., 2., 3., 10., 100., 300., 1000.]
# CelebA
REG = "l1"
# # REG = "l2"
# C_OPTIONS = [3., 1., 0.3, 0.1, 0.03, 0.01, 0.003]
# CLASS_WEIGHT_OPTIONS = [1., 2., 3., 10., 100, 300, 500]

CLASS_WEIGHT_OPTIONS = [{0: 1, 1: w} for w in CLASS_WEIGHT_OPTIONS] + [
        {0: w, 1: 1} for w in CLASS_WEIGHT_OPTIONS]


parser = argparse.ArgumentParser(description="Tune and evaluate DFR.")
parser.add_argument(
    'dataset', type=str, default='cifar10',
    help="Train dataset")
parser.add_argument(
    '--model', type=str, default='resnet18')
parser.add_argument(
    '--project_name', type=str, default='table-5-dfr')
parser.add_argument(
    '--run_name', type=str, default='')
parser.add_argument(
    '--num_workers', type=int, default=4)
parser.add_argument(
    '--no_normalize', action='store_true', help='whether to not normalize data')
parser.add_argument(
    "--result_path", type=str, default="logs/",
    help="Path to save results")
parser.add_argument(
    "--ckpt_path", type=str, default=None, help="Checkpoint path")
parser.add_argument(
    "--batch_size", type=int, default=128, required=False,
    help="Batch size")
parser.add_argument(
    "--percent_train", type=float, default=1.0, required=False,
    help="Percent of clean training data to use")
args = parser.parse_args()


def dfr_on_validation_tune(all_embeddings, all_y, preprocess=True, num_retrains=1):
    accs_dict = {}

    for i in range(num_retrains):
        x_val = all_embeddings["train"]
        y_val = all_y["train"]

        if preprocess:
            scaler = StandardScaler()
            x_val = scaler.fit_transform(x_val)

        for c in C_OPTIONS:
            logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear")
            logreg.fit(x_val, y_val)
            preds_val = logreg.predict(x_val)
            acc = (preds_val == y_val).mean()
            accs_dict[c] = acc

    ks, vs = list(accs_dict.keys()), list(accs_dict.values())
    best_hypers = ks[np.argmax(vs)]
    return best_hypers


def dfr_on_validation_eval(c, all_embeddings, all_y, num_retrains=20, preprocess=True, best_coeff=None, best_intercept=None):
    coefs, intercepts = [], []
    if preprocess:
        scaler = StandardScaler()
        scaler.fit(all_embeddings["train"])

    for i in range(num_retrains):
        x_val = all_embeddings["train"]
        y_val = all_y["train"]

        if preprocess:
            x_val = scaler.transform(x_val)

        logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear")
        logreg.fit(x_val, y_val)
        coefs.append(logreg.coef_)
        intercepts.append(logreg.intercept_)

    x_test = all_embeddings["test"]
    y_test = all_y["test"]

    if preprocess:
        x_test = scaler.transform(x_test)
    logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear")

    # the fit is only needed to set up logreg
    n_classes = 10
    logreg.fit(x_val[:n_classes], np.arange(n_classes))
    logreg.coef_ = np.mean(coefs, axis=0)
    logreg.intercept_ = np.mean(intercepts, axis=0)
    preds_test = logreg.predict(x_test)
    test_acc = (preds_test == y_test).mean()
    return test_acc

def get_model(args):
    model = ResNet18(num_classes=10)
    # d = model.linear.in_features
    if args.ckpt_path is not None and ('random-init-network' not in args.ckpt_path):
        state_dict = torch.load(os.path.join(args.ckpt_path, 'epoch=59-v1.ckpt'))['state_dict']
        # Remove 'model.' prefix
        state_dict = {k[6:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict=state_dict)
    # model.linear = torch.nn.Linear(d, 10)
    model.cuda()
    model.eval()
    return model


## Load data
train_loader = get_train_dataset('clean', args.batch_size, args.num_workers, percent_train=args.percent_train)
test_loader  = get_test_dataset(8*args.batch_size, args.num_workers, normalize=(not args.no_normalize))


# Load model
n_classes = 10
model = get_model(args)

# Evaluate model
print("Base Model")
base_model_results = {}
base_model_results["train"] = evaluate_no_min_group(model, train_loader)
base_model_results["test"] = evaluate_no_min_group(model, test_loader)
print(base_model_results)
print()
model.eval()

# Extract embeddings
def get_embed(m, x):
    out = F.relu(m.bn1(m.conv1(x)))
    out = m.layer1(out)
    out = m.layer2(out)
    out = m.layer3(out)
    out = m.layer4(out)
    out = F.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    return out

all_embeddings = {}
all_y = {}
for name, loader in [("train", train_loader), ("test", test_loader)]:
    all_embeddings[name] = []
    all_y[name] = []
    for x, y in tqdm.tqdm(loader):
        with torch.no_grad():
            all_embeddings[name].append(get_embed(model, x.cuda()).detach().cpu().numpy())
            all_y[name].append(y.detach().cpu().numpy())
    all_embeddings[name] = np.vstack(all_embeddings[name])
    all_y[name] = np.concatenate(all_y[name])


# DFR on validation
print("DFR on validation")
dfr_val_results = {}
c = dfr_on_validation_tune(all_embeddings, all_y)
dfr_val_results["best_hypers"] = c
print("Best hyperparam c:", dfr_val_results["best_hypers"])

test_acc = dfr_on_validation_eval(c, all_embeddings, all_y, num_retrains=1)
dfr_val_results["test_acc"] = test_acc
print(dfr_val_results)
print()



all_results = {}
all_results["base_model_results"] = base_model_results
all_results["dfr_val_results"] = dfr_val_results
all_results["train_embeddings_shape"] = all_embeddings["train"].shape
print(all_results)

results_path = os.path.join(args.result_path, f'percent_{args.percent_train}')
if not os.path.exists(results_path):
    os.makedirs(results_path)
with open(os.path.join(results_path, f'{args.dataset}.json'), 'w') as f:
    # dump all_results as readable json
    json.dump(all_results, f, indent=4)