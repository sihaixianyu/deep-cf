import numpy as np
import torch.utils.data as data


class BaseDataset(data.Dataset):
    def __init__(self, user_num, item_num, inter_mat):
        self.user_num = user_num
        self.item_num = item_num
        self.user_mat = inter_mat
        self.item_mat = inter_mat.T

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()


class TrainDataset(BaseDataset):
    def __init__(self, user_num, item_num, inter_mat, train_list, neg_num=4):
        super().__init__(user_num, item_num, inter_mat)
        self.train_list = train_list
        self.neg_num = neg_num

        self.train_arr = self.neg_sample()

    def __len__(self):
        return len(self.train_arr)

    def __getitem__(self, idx):
        uid = self.train_arr[idx][0]
        iid = self.train_arr[idx][1]
        rating = self.train_arr[idx][2]

        user_vec = self.user_mat[uid]
        item_vec = self.item_mat[iid]

        return user_vec, item_vec, rating

    def neg_sample(self):
        assert self.neg_num > 0, 'neg_num must be larger than 0'

        train_arr = []
        for arr in self.train_list:
            uid, pos_iid = arr[0], arr[1]
            train_arr.append([uid, pos_iid, np.float32(1)])
            for _ in range(self.neg_num):
                neg_iid = np.random.randint(self.item_num)
                while (uid, neg_iid) in self.user_mat:
                    neg_iid = np.random.randint(self.item_num)
                train_arr.append([uid, neg_iid, np.float32(0)])

        return train_arr


class TestDataset(BaseDataset):
    def __init__(self, user_num, item_num, inter_mat, test_list):
        super().__init__(user_num, item_num, inter_mat)
        self.test_list = test_list
        self.test_arr = np.array(test_list)

    def __len__(self):
        return len(self.test_arr)

    def __getitem__(self, idx):
        uid = self.test_arr[idx][0]
        iid = self.test_arr[idx][1]

        user_vec = self.user_mat[uid]
        item_vec = self.item_mat[iid]

        return uid, iid, user_vec, item_vec
