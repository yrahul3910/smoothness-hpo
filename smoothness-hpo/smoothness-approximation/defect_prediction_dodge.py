from src.util import defect_file_dic, load_defect_data

from raise_utils.hyperparams import DODGE
from raise_utils.learners import (
    DecisionTree,
    LogisticRegressionClassifier,
    NaiveBayes,
    RandomForest,
)
from raise_utils.transforms import Transform


for filename in defect_file_dic.keys():
    print(filename)
    data_orig = load_defect_data(filename)
    transform = Transform("smote")
    transform.apply(data_orig)

    config = {
        "n_runs": 10,
        "transforms": ["normalize", "standardize", "minmax", "maxabs"],
        "metrics": ["pd-pf", "pd", "pf", "auc", "f1"],
        "log_path": "dodge_logs",
        "learners": [
            LogisticRegressionClassifier(random=True, name="lr"),
            NaiveBayes(random=True, name="nb"),
            RandomForest(random=True),
            DecisionTree(random=True, name="dt"),
        ]
        * 10,
        "n_iters": 30,
        "name": f"defect-{filename}",
        "data": [data_orig],
    }
    dodge = DODGE(config)
    perfs = dodge.optimize()

    print(perfs)
    print()
