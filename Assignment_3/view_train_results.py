import pandas as pd
import os
import argparse
from typing import List
import numpy as np
import json
import re

def file_search(src_path: str, tgt_file: str):

    def recurse(tgt_path: str, tgt_file: str, output_list: List):

        if os.path.isdir(tgt_path):
            for file in os.listdir(tgt_path):
                recurse(os.path.join(tgt_path, file), tgt_file, output_list)
        else:
            if os.path.basename(tgt_path) == tgt_file:
                output_list.append(tgt_path)

    output_list = list()

    recurse(src_path, tgt_file, output_list)
    
    return output_list

def reduce_to_best_models(df):

    if 'config' not in df.columns or 'metrics/mAP50-95(B)' not in df.columns:
        raise ValueError("DataFrame must contain 'config' and 'metrics/mAP50-95(B)' columns")
    
    # Select the row with the highest 'mAP50-95(B)' for each unique 'config'
    df['config'] = df['config'].astype(str)
    best_models = df.loc[df.groupby('config')['metrics/mAP50-95(B)'].idxmax()]
    best_models = best_models.sort_values(by='metrics/mAP50-95(B)', ascending=False)

    return best_models

def main():

    parser = argparse.ArgumentParser(
        'view_raytune_results',
    )

    parser.add_argument(
        'source',
    )

    src_directory = parser.parse_args().source
    target_file = 'result.json'

    paths = file_search(src_directory, target_file)

    results = pd.DataFrame()

    for file in paths:
        with open(file, 'r') as f:
            try:
                data = [json.loads(line) for line in f]
                df = pd.DataFrame(data)

                results = pd.concat([results, df], ignore_index=True)
            except Exception as E:
                print(E)
    
    results = reduce_to_best_models(results)
    results.to_csv('results.csv')

if __name__ == "__main__":
    main()