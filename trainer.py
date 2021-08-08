import sys

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CFNet
import torch.nn.functional as F


class Trainer:
    def __init__(self, loader: DataLoader, model: CFNet, optimizer: Optimizer):
        self.loader = loader
        self.model = model
        self.optimizer = optimizer

    def train(self) -> float:
        self.model.train()

        loss_list = []
        for uids, iids, user_vecs, item_vecs, ratings in tqdm(self.loader, desc='Train', file=sys.stdout):
            self.optimizer.zero_grad()
            uids = uids.to(self.model.device)
            iids = iids.to(self.model.device)
            user_vecs = user_vecs.to(self.model.device)
            item_vecs = item_vecs.to(self.model.device)
            real_ratings = ratings.to(self.model.device)
            pred_ratings = self.model.forward(uids, iids, user_vecs, item_vecs)

            loss = F.binary_cross_entropy(torch.squeeze(pred_ratings), real_ratings)
            loss.backward()
            loss_list.append(loss.detach().cpu().numpy())
            self.optimizer.step()

        return np.mean(loss_list).item()
