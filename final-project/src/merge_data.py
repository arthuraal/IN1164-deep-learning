import os

import pandas as pd
from tqdm.auto import tqdm

COT_PATH = '../data/cot_data'

file_list = os.listdir(COT_PATH)
final_data = pd.DataFrame()
for f_name in tqdm(file_list, total=len(file_list), desc='Merging files'):
    data = pd.read_csv(os.path.join(COT_PATH, f_name), sep='\t', header=None)
    data.columns = ['question', 'answer', 'rationale']
    final_data = pd.concat([final_data, data])

# Saving new data
final_data.to_csv('../data/cot_train_data.csv', header=True, index=False)
