import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, user_num, item_num, mlp_dims, device='cpu'):
        super(MLP, self).__init__()

        # In the official implement, the first layer has no activation
        self.embed_user = nn.Linear(item_num, mlp_dims[0] // 2)
        self.embed_item = nn.Linear(user_num, mlp_dims[0] // 2)

        self.mlp = nn.Sequential()
        for i in range(len(mlp_dims) - 1):
            self.mlp.add_module('mlp_linear_%d' % i, nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            self.mlp.add_module('mlp_relu_%d' % i, nn.ReLU())

        self.pred_layer = nn.Sequential(
            nn.Linear(mlp_dims[-1], 1),
            nn.Sigmoid(),
        )

        self.device = device
        self.to(device)

    def forward(self, user_vecs, item_vecs):
        user_embedding = self.embed_user(user_vecs)
        item_embedding = self.embed_item(item_vecs)
        mlp_vecs = torch.cat((user_embedding, item_embedding), dim=1)
        mlp_vecs = self.mlp(mlp_vecs)

        pred_ratings = self.pred_layer(mlp_vecs)
        return pred_ratings

    def predict(self, user_vecs, item_vecs):
        with torch.no_grad():
            pred_ratings = self.forward(user_vecs, item_vecs)

        return pred_ratings
