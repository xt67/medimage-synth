#!/usr/bin/env python3
import yaml
import torch
import json
import time
import numpy as np

from src.data_loader import create_dataloaders
from src.classifier.train_cls import train_classifier, evaluate_classifier
from src.utils.seed import set_seed


def main():
    cfg = yaml.safe_load(open('configs/classifier.yaml'))
    cfg['epochs'] = 1
    cfg['pretrained'] = True
    cfg['batch_size'] = 16

    tmp_cfg = 'C:/Users/onlys/AppData/Local/Temp/opencode/cfg/cb1.yaml'
    with open(tmp_cfg, 'w') as f:
        yaml.dump(cfg, f)

    tr, vl, te = create_dataloaders(
        cfg['data_dir'], batch_size=16, image_size=128, num_workers=0
    )

    results = []
    for seed in [42, 123, 999]:
        set_seed(seed)
        t0 = time.time()
        m = train_classifier(
            tmp_cfg, tr, vl, torch.device('cpu'),
            output_dir=f'outputs/baseline_seed{seed}_1ep'
        )
        t1 = time.time()
        met = evaluate_classifier(m, te, torch.device('cpu'))
        met['seed'] = seed
        met['time_sec'] = t1 - t0
        results.append(met)
        print('Seed {}: Recall={:.4f}, F1={:.4f}, Time={:.1f}s'.format(
            seed, met['recall'], met['f1'], t1 - t0))

    print('\n--- SUMMARY ---')
    for k in ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'roc_auc', 'auprc']:
        vals = [r[k] for r in results if r[k] is not None]
        print('{:12s}: {:.4f} ± {:.4f}'.format(k, np.mean(vals), np.std(vals)))

    with open('outputs/baseline_3seeds_1ep.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()