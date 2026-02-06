import torch
import numpy as np
import argparse
import os
from scipy import io
import importlib
from sklearn.cluster import KMeans
import glob
from DataSet.DataLoader import get_dataloader
import ot
from utils import aucPerformance, get_logger, F1Performance
import ipdb
import time
import random
from adbench.baseline.PyOD import PYOD
from adbench.baseline.DAGMM.run import DAGMM
from sklearn.cluster import KMeans

DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data')
npz_files = glob.glob(os.path.join(DATA_DIR, '*.npz'))
npz_datanames = [os.path.splitext(os.path.basename(file))[0] for file in npz_files]

mat_files = glob.glob(os.path.join(DATA_DIR, '*.mat'))
mat_datanames = [os.path.splitext(os.path.basename(file))[0] for file in mat_files]

def train_test_split(inliers, outliers, model_config):
    num_split = len(inliers) // 2
    train_data = inliers[:num_split]
    train_label = np.zeros(num_split)
    test_data = np.concatenate([inliers[num_split:], outliers], 0)

    test_label = np.zeros(test_data.shape[0])
    test_label[-len(outliers):] = 1

    return train_data, train_label, test_data, test_label


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--preprocess', type=str, default='standard')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epoch', type=int, default=200)
    # the number of k means centers, i.e., the number of point in Q
    parser.add_argument('--n_clusters', type=int, default=5)
    # the number of point in distribution P
    parser.add_argument('--sub_train', type=int, default=20)
    parser.add_argument('--calib_lambda', type=float, default=1.0)
    args = parser.parse_args()

    dict_to_import = 'model_config_'+args.model_type
    module_name = 'configs'
    module = importlib.import_module(module_name)
    model_config = getattr(module, dict_to_import)

    model_config['preprocess'] = args.preprocess
    model_config['random_seed'] = args.seed
    model_config['epochs'] = args.epoch

    torch.manual_seed(model_config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(model_config['random_seed'])
    random.seed(model_config['random_seed'])
    np.random.seed(model_config['random_seed'])
    if model_config['num_workers'] > 0:
        torch.multiprocessing.set_start_method('spawn')

    data_dir = os.path.join(os.path.dirname(__file__), model_config['data_dir'])
    if args.dataname in npz_datanames:
        path = os.path.join(data_dir, args.dataname + '.npz')
        data = np.load(path)
    elif args.dataname in mat_datanames:
        path = os.path.join(data_dir, args.dataname + '.mat')
        data = io.loadmat(path)
    else:
        available = sorted(npz_datanames + mat_datanames)
        raise FileNotFoundError(
            f"Dataset '{args.dataname}' not found in {data_dir}. Available: {available}"
        )
    samples = data['X']
    model_config['dataset_name'] = args.dataname
    model_config['data_dim'] = samples.shape[-1]

    labels = ((data['y']).astype(int)).reshape(-1)
    inliers = samples[labels == 0]
    outliers = samples[labels == 1]
    train_data, train_label, test_data, test_label = train_test_split(inliers, outliers, model_config)
    # Guard against NaN/Inf values that can break PCA (and KMeans/OT).
    train_data = np.nan_to_num(train_data, nan=0.0, posinf=0.0, neginf=0.0)
    test_data = np.nan_to_num(test_data, nan=0.0, posinf=0.0, neginf=0.0)


    model = PYOD(seed=args.seed, model_name=args.model_type)  # initialization
    model.fit(train_data, train_label)  # fit
    score = model.predict_score(test_data)  # predict
    # Some models (e.g., PCA) can yield inf/NaN scores; sanitize before metrics.
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(open(f'./results/{args.dataname}_{args.model_type}_test_score.npy','wb'), score)
    mse_rauc, mse_ap = aucPerformance(score, test_label)
    mse_f1 = F1Performance(score, test_label)

    print('##########################################################################')
    print("Baseline AUC-ROC: %.4f  AUC-PR: %.4f"
          % (mse_rauc, mse_ap))
    print("f1: %.4f" % (mse_f1))

    results_dict = {'AUC-ROC':mse_rauc, 'AUC-PR':mse_ap, 'f1':mse_f1}
    np.save(open(f'./results/{args.dataname}_{args.model_type}_results.npy','wb'), results_dict)

    kmeans = KMeans(n_clusters=args.n_clusters)
    kmeans.fit(train_data)
    # distribution Q
    random_samples = kmeans.cluster_centers_

    # distribution P
    random_samples_tilde = train_data[np.random.choice(train_data.shape[0], args.sub_train, replace=False)]

    score_ = []
    for i in range(test_data.shape[0]):
        # plus one test point to distribution P
        construct = np.concatenate([random_samples_tilde, test_data[i,:].reshape(1,-1)], axis=0)

        M = np.linalg.norm(construct[:, None] - random_samples, axis=-1)
        u = np.ones(construct.shape[0]) / construct.shape[0] 
        v = np.ones(args.n_clusters) / args.n_clusters
        score_.append((ot.emd(u, v, M) * M).mean(axis=1)[-1])
    
    score_ = np.array(score_)
    score_ = np.nan_to_num(score_, nan=0.0, posinf=0.0, neginf=0.0)

    np.save(open(f'./results/{args.dataname}_{args.model_type}_{args.n_clusters}_{args.sub_train}_lam{args.calib_lambda}_ot_distance.npy','wb'), score_)

    base_norm = (score - score.min()) / (score.max() - score.min() + 1e-12)
    ot_norm = (score_ - score_.min()) / (score_.max() - score_.min() + 1e-12)
    score = base_norm + (args.calib_lambda * ot_norm)

    mse_rauc, mse_ap = aucPerformance(score, test_label)
    mse_f1 = F1Performance(score, test_label)

    print('##########################################################################')
    print("AUC-ROC: %.4f  AUC-PR: %.4f"
          % (mse_rauc, mse_ap))
    print("f1: %.4f" % (mse_f1))

    results_dict = {'AUC-ROC':mse_rauc, 'AUC-PR':mse_ap, 'f1':mse_f1}
    np.save(open(f'./results/{args.dataname}_{args.model_type}_{args.n_clusters}_{args.sub_train}_lam{args.calib_lambda}_OTAD.npy','wb'), results_dict)
