"""
    Step0_DATA.PY is to sample the data to 10W
"""
import os
import pandas as pd

SAMPLE_NUM = 90000 # sample num 10w
TICHI_TRAIN_FILE = './data/round2_train.txt'
SAMPLE_TRAIN_FILE = 'sample_train.csv'


if not os.path.exists(SAMPLE_TRAIN_FILE):
    df = pd.read_csv(TICHI_TRAIN_FILE, sep=' ')
    df = df.sample(n = SAMPLE_NUM)
    with open(SAMPLE_TRAIN_FILE, 'w') as file:
        df.to_csv(file, index=False)

else:
    print("The file " + SAMPLE_TRAIN_FILE + " exists")
