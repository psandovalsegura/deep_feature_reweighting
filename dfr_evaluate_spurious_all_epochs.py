import os
import json
import argparse
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from poison_datasets import get_train_dataset, get_test_dataset
from utils import evaluate_no_min_group
from models import *

REG = "l1"

def get_model(ckpt_path):
    model = ResNet18(num_classes=10)
    # d = model.linear.in_features
    if ckpt_path is not None and ('random-init-network' not in ckpt_path):
        state_dict = torch.load(ckpt_path)['state_dict']
        # Remove 'model.' prefix
        state_dict = {k[6:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict=state_dict)
    # model.linear = torch.nn.Linear(d, 10)
    model.cuda()
    model.eval()
    return model

def get_embed(m, x):
    out = F.relu(m.bn1(m.conv1(x)))
    out = m.layer1(out)
    out = m.layer2(out)
    out = m.layer3(out)
    out = m.layer4(out)
    out = F.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    return out

def get_all_embeddings(model, train_loader, test_loader):
    all_embeddings = {}
    all_y = {}
    for name, loader in [("train", train_loader), ("test", test_loader)]:
        all_embeddings[name] = []
        all_y[name] = []
        for x, y in loader:
            with torch.no_grad():
                all_embeddings[name].append(get_embed(model, x.cuda()).detach().cpu().numpy())
                all_y[name].append(y.detach().cpu().numpy())
        all_embeddings[name] = np.vstack(all_embeddings[name])
        all_y[name] = np.concatenate(all_y[name])
    return all_embeddings, all_y

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

def main():
    parser = argparse.ArgumentParser(description="Tune and evaluate DFR on all checkpoints.")
    parser.add_argument(
        'ckpt_directory', type=str, default='',
        help="The directory containing the checkpoints")
    parser.add_argument(
        "--result_path", type=str, default="logs/",
        help="Path to save results")
    parser.add_argument(
        '--num_workers', type=int, default=4)
    parser.add_argument(
        "--batch_size", type=int, default=128, required=False,
        help="Batch size")
    parser.add_argument(
        "--percent_train", type=float, default=0.3, required=False,
        help="Percent of clean training data to use")
    args = parser.parse_args()
    print(args)

    ## Load data (30% of original training set)
    train_loader = get_train_dataset('clean', 8*args.batch_size, args.num_workers, percent_train=args.percent_train)
    test_loader  = get_test_dataset(8*args.batch_size, args.num_workers, normalize=True)
    
    # Get the list of checkpoints
    ckpt_list = os.listdir(args.ckpt_directory)
    ckpt_list = [os.path.join(args.ckpt_directory, ckpt) for ckpt in ckpt_list]

    dfr_results = {}
    for ckpt in ckpt_list:
        # Tune and evaluate DFR on this checkpoint
        print("Evaluating DFR on checkpoint", ckpt)
        model = get_model(ckpt)

        # Evaluate model
        base_model_results = {}
        epoch = int(ckpt.split("epoch=")[1].split(".ckpt")[0])
        base_model_results["base_test"] = evaluate_no_min_group(model, test_loader)
        model.eval()

        all_embeddings, all_y = get_all_embeddings(model, train_loader, test_loader)

        c = 1.0
        test_acc = dfr_on_validation_eval(c, all_embeddings, all_y, num_retrains=1)
        base_model_results["dfr_test"] = test_acc
        dfr_results[epoch] = base_model_results

    # Print results in the following format:
    # Base test accuracies: (epoch, base_test)
    # (epoch, base_test)
    # (epoch, base_test)
    # ...
    # DFR test accuracies: (epoch, dfr_test)
    # (epoch, dfr_test)
    # (epoch, dfr_test)
    # ...
    max_epoch = max(dfr_results.keys())
    print("Base test accuracies:")
    for i in range(max_epoch+1):
        print(f"({dfr_results[i]['epoch']}, {dfr_results[i]['base_test']:0.3f})")
    print("DFR test accuracies:")
    for i in range(max_epoch+1):
        print(f"({dfr_results[i]['epoch']}, {dfr_results[i]['dfr_test']:0.3f})")

    # Save results
    dataset_name = args.ckpt_directory.split("every-epoch-ckpt/linf-poison/")[1]
    results_path = os.path.join(args.result_path, 'every-epoch-ckpt', dataset_name)
    print('results_path', results_path)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    with open(os.path.join(results_path, f'{dataset_name}.json'), 'w') as f:
        # dump all_results as readable json
        json.dump(dfr_results, f, indent=4)

if __name__ == '__main__':
    main()