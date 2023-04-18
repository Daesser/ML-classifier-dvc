import os.path
import sweetviz as sv
import pandas as pd
import yaml

"""
This script creates a report using sweetviz. Creates report in reports folder.
"""

data = yaml.safe_load(open('params.yaml'))['data']
path_raw = data["dataset_raw"]  # Root directory of raw data
path_processed = data["dataset_processed"]  # Root directory of raw data
datasets = data["datasets"]

if __name__ == '__main__':
    for dataset in datasets:
        train_path = os.path.join(path_processed, dataset, 'tags_train.csv')
        train_df = pd.read_csv(train_path)
        val_path = os.path.join(path_processed, dataset, 'tags_val.csv')
        val_df = pd.read_csv(val_path)
        train_df['class'] = pd.factorize(train_df['class'])[0]
        val_df['class'] = pd.factorize(val_df['class'])[0]
        feature_config = sv.FeatureConfig(skip="filename", force_num=["class"])
        comparison_report = sv.compare([train_df, 'Train'], [val_df, 'Validation'], None, feature_config)
        comparison_report.show_html(f'reports/Pretrain_report_{dataset}.html')

    # Dummy file. See https://github.com/iterative/dvc/issues/8881. May be resolved in future
    f = open("temp/dummy_analyze.txt", "w")
    f.write("Analysis complete")
    f.close()
