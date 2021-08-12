import sys

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm


class Evaluator:
    def __init__(self, loader: DataLoader, model: Module, topk=10):
        self.loader = loader
        self.model = model
        self.topk = topk

    def evaluate(self) -> (float, float):
        self.model.eval()

        hit_list, ndcg_list = [], []
        for _, iids, user_vecs, item_vecs in tqdm(self.loader, desc='Evaluate', file=sys.stdout):
            iids = iids.to(self.model.device)
            user_vecs = user_vecs.to(self.model.device)
            item_vecs = item_vecs.to(self.model.device)

            pred_ratings = self.model.predict(user_vecs, item_vecs)
            pred_ratings = torch.squeeze(pred_ratings)
            _, idxs = torch.topk(pred_ratings, self.topk)
            rec_list = torch.take(iids, idxs).cpu().numpy().tolist()

            pos_item = iids[0].item()
            hit_list.append(self.hit(pos_item, rec_list))
            ndcg_list.append(self.ndcg(pos_item, rec_list))

        return np.mean(hit_list).item(), np.mean(ndcg_list).item()

    @staticmethod
    def hit(iid, rec_list) -> float:
        if iid in rec_list:
            return 1
        return 0

    @staticmethod
    def ndcg(iid, rec_list) -> float:
        if iid in rec_list:
            idx = rec_list.index(iid)
            return np.reciprocal(np.log2(idx + 2))
        return 0
