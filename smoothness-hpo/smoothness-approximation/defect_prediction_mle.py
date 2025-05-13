import os
import random

from copy import deepcopy

from src.util import (
    defect_file_dic,
    get_many_random_hyperparams,
    get_smoothness_mle_approx,
    load_defect_data,
)

import numpy as np
import pandas as pd

from raise_utils.metrics import ClassificationMetrics
from raise_utils.transforms import Transform
from sklearn.tree import DecisionTreeClassifier


hpo_space = {
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": list(range(1, 11)),
    "min_samples_split": [2, 3, 4, 5],
    "max_features": ["sqrt", "log2", 2, 3, 4, 5],
    "transform": ["normalize", "standardize", "minmax", "maxabs"]
}


file_number = os.getenv('SLURM_JOB_ID') or random.randint(1, 10000)
full_results = {"dataset": [], "pd-pf": [], "pd": [], "pf": [], "f1": [], "auc": []}
for filename in defect_file_dic.keys():
    file = open(f'runs-{file_number}.txt', 'a')
    print(f'{filename}:', file=file)

    data_orig = load_defect_data(filename)

    best_betas = []
    best_configs = []
    keep_configs = 10
    num_configs = 50

    configs = get_many_random_hyperparams(hpo_space, num_configs)

    for config in configs:
        data = deepcopy(data_orig)
        data.x_train = np.array(data.x_train)
        data.y_train = np.array(data.y_train)

        transform_name = config.pop("transform")
        transform = Transform(transform_name)
        transform.apply(data)

        model = DecisionTreeClassifier(**config)
        model.fit(data.x_train, data.y_train)
        smoothness = get_smoothness_mle_approx(data, model)
        file.flush()

        if len(best_betas) < keep_configs or smoothness > min(best_betas):
            best_betas.append(smoothness)
            best_configs.append(config)

            best_betas, best_configs = zip(*sorted(zip(best_betas, best_configs, strict=False), reverse=True, key=lambda x: x[0]), strict=False)
            best_betas = list(best_betas[:keep_configs])
            best_configs = list(best_configs[:keep_configs])

    best_metrics = [0., 0., 0., 0., 0.]
    for beta, config in zip(best_betas, best_configs, strict=False):
        data = deepcopy(data_orig)
        model = DecisionTreeClassifier(**config)
        model.fit(data.x_train, data.y_train)
        preds = model.predict(data.x_test)

        metrics = ClassificationMetrics(data.y_test, preds)
        metrics.add_metrics(['pd-pf', 'pd', 'pf', 'f1', 'auc'])
        results = metrics.get_metrics()
        print(f'Config: {config} | Beta: {beta} | Metrics: {results}')

        if results[0] > best_metrics[0]:
            best_metrics = results[:]

    print(f"Best metrics: {best_metrics}")
    full_results['dataset'] += [filename]
    full_results['pd-pf'] += [best_metrics[0] * 100.]
    full_results['pd'] += [best_metrics[1] * 100.]
    full_results['pf'] += [best_metrics[2] * 100.]
    full_results['f1'] += [best_metrics[3] * 100.]
    full_results['auc'] += [best_metrics[4] * 100.]

df = pd.DataFrame(full_results)
df.to_csv("defect.csv", float_format="%.1f")

print(df[["dataset", "pd", "pf", "auc"]].round(1))
