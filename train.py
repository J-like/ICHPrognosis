import argparse
import torch
import random
import numpy as np
import os
from datetime import datetime
from torch.utils.data import SubsetRandomSampler, DataLoader
from sklearn.model_selection import StratifiedKFold
from data.data_to_dataset import dataset3D
from data.read_excel import screch_excel
from net.run_model import run
def init_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_to_train(batch_size, epochs, learning_rate):
    device = torch.device("cuda:0")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    patients_info = screch_excel('./samples.xlsx')
    datasets = dataset3D(patients_info)
    labels = [datasets[i][2] for i in range(len(datasets))]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_ids, val_ids) in enumerate(skf.split(datasets, labels)):
        print(f"FOLD {fold+1} | Train samples: {len(train_ids)}, Val samples: {len(val_ids)}")

        train_loader = DataLoader(datasets, batch_size=batch_size, sampler=SubsetRandomSampler(train_ids), drop_last=True, num_workers=0)
        val_loader   = DataLoader(datasets, batch_size=1, sampler=SubsetRandomSampler(val_ids), num_workers=0)

        run(device=device, epochs=epochs, train_loader=train_loader, val_loader=val_loader,
                train_count=len(train_ids), val_count=len(val_ids), fold=fold, learning_rate=learning_rate)

if __name__ == "__main__":
    init_seeds(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    args = parser.parse_args()

    prepare_to_train(batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.learning_rate)
