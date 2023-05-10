import os
import json
import argparse
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from poison_datasets import construct_train_dataset, get_test_loader
from utils import evaluate_no_min_group
from models import *

REG = "l1"

def get_model(ckpt_path):
    model = ResNet18(num_classes=10)
    # d = model.linear.in_features
    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path)
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

def print_and_save(dfr_results, ckpt_directory, result_path):
    # Print results in the following format:
    # Base test accuracies: (epoch, pre_dfr)
    # (epoch, pre_dfr)
    # (epoch, pre_dfr)
    # ...
    # DFR test accuracies: (epoch, post_dfr)
    # (epoch, post_dfr)
    # (epoch, post_dfr)
    # ...
    max_epoch = max(dfr_results.keys())
    print("CKPT test accuracies:")
    for i in range(max_epoch+1):
        print(f"({i}, {dfr_results[i]['pre_dfr'] * 100:0.3f})")
    print("DFR test accuracies:")
    for i in range(max_epoch+1):
        print(f"({i}, {dfr_results[i]['post_dfr'] * 100:0.3f})")

    # Save results
    ckpt_name = ckpt_directory.split("/")[-1]
    results_path = os.path.join(result_path, ckpt_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Save results as list for easy reading
    text_path = os.path.join(results_path, f'list.txt')
    with open(text_path, 'w') as f:
        f.write("Base test accuracies:\n")
        for i in range(max_epoch+1):
            f.write(f"({i}, {dfr_results[i]['pre_dfr'] * 100:0.3f})\n")
        f.write("DFR test accuracies:\n")
        for i in range(max_epoch+1):
            f.write(f"({i}, {dfr_results[i]['post_dfr'] * 100:0.3f})\n")

    # Save results as json for easy loading
    json_path = os.path.join(results_path, f'list.json')
    with open(json_path, 'w') as f:
        # dump all_results as readable json
        json.dump(dfr_results, f, indent=4)

def get_subset_train_loader(setup_key, percent, batch_size, num_workers, normalize=True):
    train_ds = construct_train_dataset(setup_key=setup_key,
                                       normalize=normalize)
    train_idx = np.load(os.path.join('dfr_indices', f"train_{percent}pct.npy"))
    train_ds = torch.utils.data.Subset(train_ds, train_idx)
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=True)
    return train_loader



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
        "--percent_train", type=int, default=10, required=False,
        help="Percent of clean training data to use")
    parser.add_argument(
        "--random_init_ckpt", action="store_true",
        help="Whether to use random init ckpt"
    )
    args = parser.parse_args()
    print(args)

    args.result_path = os.path.join(args.result_path, f'{args.percent_train}pct')

    ## Load data 
    train_loader = get_subset_train_loader(setup_key='cifar10',
                                           batch_size=8*args.batch_size,
                                           num_workers=args.num_workers,
                                           percent=args.percent_train)
                
    test_loader  = get_test_loader(setup_key='cifar10', 
                                   batch_size=8*args.batch_size, 
                                   num_workers=args.num_workers, 
                                   normalize=True)

    # If random init ckpt, then compute results for only one model
    if args.random_init_ckpt:
        print("Evaluating DFR on random init model")
        dfr_results = {}
        model = get_model(None)
        base_model_results = {}
        base_model_results["pre_dfr"] = evaluate_no_min_group(model, test_loader)

        all_embeddings, all_y = get_all_embeddings(model, train_loader, test_loader)
        c = 1.0
        test_acc = dfr_on_validation_eval(c, all_embeddings, all_y, num_retrains=1)
        base_model_results["post_dfr"] = test_acc
        dfr_results[0] = base_model_results
        print_and_save(dfr_results, 'tmp/random-init', args.result_path)
        return
    
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
        epoch = int(ckpt.split("=")[-1].split(".")[0]) # checkpoint name is of the form 'epoch={e}.pt'
        base_model_results["pre_dfr"] = evaluate_no_min_group(model, test_loader)

        all_embeddings, all_y = get_all_embeddings(model, train_loader, test_loader)

        c = 1.0
        test_acc = dfr_on_validation_eval(c, all_embeddings, all_y, num_retrains=1)
        base_model_results["post_dfr"] = test_acc
        dfr_results[epoch] = base_model_results
    print_and_save(dfr_results, args.ckpt_directory, args.result_path)

if __name__ == '__main__':
    main()