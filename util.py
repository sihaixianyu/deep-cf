import numpy as np
import pandas as pd


def load_data(data_prefix: str) -> (int, int, np.ndarray, list, list):
    train_df = pd.read_csv(
        '{}.train.rating'.format(data_prefix),
        sep='\t', header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    user_num = train_df['user'].max() + 1
    item_num = train_df['item'].max() + 1

    train_list = train_df.values.tolist()
    inter_mat = np.zeros((user_num, item_num), dtype=np.float32)
    for x in train_list:
        inter_mat[x[0], x[1]] = 1.0

    test_list = []
    with open('{}.test.negative'.format(data_prefix), 'r') as fd:
        line = fd.readline()
        while line and line != '':
            arr = line.split('\t')
            # Add latest pos_iid
            uid = eval(arr[0])[0]
            pos_iid = eval(arr[0])[1]
            test_list.append([uid, pos_iid])
            # Add random sampled 100 neg_iid
            for i in arr[1:]:
                test_list.append([uid, int(i)])
            line = fd.readline()

    return user_num, item_num, inter_mat, train_list, test_list


def print_res(content: str):
    print('-' * 100)
    print(content)
    print('-' * 100)


if __name__ == '__main__':
    pass
