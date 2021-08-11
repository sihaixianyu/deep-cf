import numpy as np
import torch.utils.data as data


class BaseDataset(data.Dataset):
    def __init__(self, info_dict: dict):
        self.user_num = info_dict['user_num']
        self.item_num = info_dict['item_num']

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()


class TrainDataset(BaseDataset):
    def __init__(self, info_dict: dict, pos_train_arr: np.ndarray, inter_mat: np.ndarray,  neg_num=4):
        super().__init__(info_dict)
        self.pos_train_arr = pos_train_arr
        self.inter_mat = inter_mat
        self.neg_num = neg_num

        self.train_arr = self.neg_sample()

    def __len__(self):
        return len(self.train_arr)

    def __getitem__(self, idx):
        uid = self.train_arr[idx][0]
        iid = self.train_arr[idx][1]
        rating = self.train_arr[idx][2]

        user_vec = self.inter_mat[uid]
        item_vec = self.inter_mat.T[iid]

        return user_vec, item_vec, rating

    def neg_sample(self):
        assert self.neg_num > 0, 'neg_num must be larger than 0'

        train_arr = []
        for arr in self.pos_train_arr:
            uid, pos_iid = arr[0], arr[1]
            train_arr.append([uid, pos_iid, np.float32(1)])
            for _ in range(self.neg_num):
                neg_iid = np.random.randint(self.item_num)
                while (uid, neg_iid) in self.inter_mat:
                    neg_iid = np.random.randint(self.item_num)
                train_arr.append([uid, neg_iid, np.float32(0)])

        return train_arr


class TestDataset(BaseDataset):
    def __init__(self, info_dict: dict, pos_test_arr: np.ndarray, inter_mat: np.ndarray, neg_dict: dict):
        super().__init__(info_dict)
        self.pos_test_arr = pos_test_arr
        self.inter_mat = inter_mat
        self.neg_dict = neg_dict

        self.test_arr = self.build_test_arr()

    def __len__(self):
        return len(self.test_arr)

    def __getitem__(self, idx):
        uid = self.test_arr[idx][0]
        iid = self.test_arr[idx][1]

        user_vec = self.inter_mat[uid]
        item_vec = self.inter_mat.T[iid]

        return uid, iid, user_vec, item_vec

    def build_test_arr(self):
        test_arr = []
        for arr in self.pos_test_arr:
            uid = arr[0]
            pos_iid = arr[1]
            test_arr.append([uid, pos_iid])
            for neg_iid in self.neg_dict[uid]:
                test_arr.append([uid, neg_iid])

        return test_arr
