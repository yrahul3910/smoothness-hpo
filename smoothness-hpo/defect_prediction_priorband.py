import os
import random
import time

from copy import deepcopy

from src.util import defect_file_dic, load_defect_data, run_experiment

import neps


hpo_space = {
    'n_units': neps.Integer(2, 6, default=4, default_confidence="low"),
    'n_layers': neps.Integer(2, 5, default=3, default_confidence="low"),
    'preprocessor': neps.Categorical(
        ['normalize', 'standardize', 'minmax', 'maxabs', 'robust'],
        default="normalize",
        default_confidence="low"
    ),
    'wfo': neps.Categorical([False, True], default=False, default_confidence="low"),
    'smote': neps.Categorical([False, True], default=False, default_confidence="low"),
    'ultrasample': neps.Categorical([False, True], default=False, default_confidence="low"),
    'smooth': neps.Categorical([False, True], default=False, default_confidence="low"),
}

file_number = os.getenv('SLURM_JOB_ID') or random.randint(1, 10000)
for filename in defect_file_dic:
    file = open(f'runs-{file_number}.txt', 'a')
    print(f'{filename}:', file=file, flush=True)

    data_orig = load_defect_data(filename)

    def objective(**config) -> dict:
        start = time.time()
        data = deepcopy(data_orig)
        results = run_experiment(data, 2, **config)
        end = time.time()

        print(f"Result: {results} | Time: {end - start}", file=file, flush=True)
        f1 = results[0]

        return {
            "loss": -f1,
            "info_dict": {
                "test_score": f1,
                "train_time": end - start
            }
        }

    neps.run(
        run_pipeline=objective,
        pipeline_space=hpo_space,
        root_directory=f"priorband_{filename}",
        max_evaluations_total=30
    )
