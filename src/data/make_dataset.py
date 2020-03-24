import os
import shutil

import numpy as np
import pickle as pkl
import pandas as pd

np.random.seed(123)

TRAIN_RATIO = 0.75

PAR_CSV = os.path.join(
    os.path.dirname(__file__),
    '../../data/external/patientid_cellmapping_parasitized.csv')

UN_CSV = os.path.join(
    os.path.dirname(__file__),
    '../../data/external/patientid_cellmapping_uninfected.csv')

TARGET_ROOT = os.path.join(
    os.path.dirname(__file__),
    '../../data/processed')

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    '../../data/raw/cell_images')

def extract_c_code(string):
    """Function to extract number following C in string (see dataset
    convention). """
    if not string[0] == 'C':
        raise Exception('First character is not C.')
    if not string[1].isdigit():
        raise Exception('Second character is not a digit.')

    i = 2
    c = string[1]
    while string[i].isdigit():
        c += string[i]
        i += 1

    return int(c)


# Prep directories
# os.mkdir(TARGET_ROOT + '/train')
# os.mkdir(TARGET_ROOT + '/test')
# os.mkdir(TARGET_ROOT + '/train' + '/X')
# os.mkdir(TARGET_ROOT + '/test' + '/X')

print('Building dataset and splits...')

df_par = pd.read_csv(PAR_CSV, header=None, index_col=0)
df_un = pd.read_csv(UN_CSV, header=None, index_col=0)

par_patients = np.unique([extract_c_code(s) for s in df_par.index.values])
un_patients_raw = np.unique([extract_c_code(s) for s in df_un.index.values])

# Isolate subjects who are only uninfected
un_patients = un_patients_raw[
    np.logical_not(np.isin(un_patients_raw, par_patients))
]

# Shuffle
np.random.shuffle(par_patients)
np.random.shuffle(un_patients)

# Splits
par_split = round(par_patients.shape[0] * TRAIN_RATIO)
un_split = round(un_patients.shape[0] * TRAIN_RATIO)

x_train = np.concatenate((par_patients[0:par_split], un_patients[0:un_split]))

x_test = np.concatenate((par_patients[par_split:], un_patients[un_split:]))

train_list = {
    'class': [],
    'id': []
}

test_list = {
    'class': [],
    'id': []
}

counter = {
    'train': 0,
    'test': 0
}


def dispatch(source_dir, class_no):
    for f in os.listdir('{}/{}'.format(DATA_PATH, source_dir)):
        if f.endswith('.png'):
            c = extract_c_code(f)

            if c in x_train:
                train_list['class'].append(class_no)
                train_list['id'].append(c)
                shutil.copy2('{}/{}/{}'.format(DATA_PATH, source_dir, f),
                             '{}/train/X/{}.png'.format(TARGET_ROOT, counter['train']))

                counter['train'] += 1

            if c in x_test:
                test_list['class'].append(class_no)
                test_list['id'].append(c)
                shutil.copy2('{}/{}/{}'.format(DATA_PATH, source_dir, f),
                             '{}/test/X/{}.png'.format(TARGET_ROOT, counter['test']))

                counter['test'] += 1


dispatch('Uninfected', 0)
dispatch('Parasitized', 1)

np.save(TARGET_ROOT + '/train/y.npy', train_list['class'])
np.save(TARGET_ROOT + '/train/subject_id.npy', train_list['id'])
np.save(TARGET_ROOT + '/test/y.npy', test_list['class'])
np.save(TARGET_ROOT + '/test/subject_id.npy', test_list['id'])
