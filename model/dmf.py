import torch
import torch.nn as nn


class DMF(nn.Module):
    def __init__(self, user_num, item_num, user_mlp_dims, item_mlp_dims, device='cpu'):
        super(DMF, self).__init__()
        assert user_mlp_dims[-1] == item_mlp_dims[-1], 'The DMF outlayer must be the same'

        user_mlp_dims.insert(0, item_num)
        item_mlp_dims.insert(0, user_num)

        self.user_mlp = nn.Sequential()
        self.item_mlp = nn.Sequential()

        for i in range(len(user_mlp_dims) - 1):
            self.user_mlp.add_module('user_linear_%d' % i, nn.Linear(user_mlp_dims[i], user_mlp_dims[i + 1]))
            self.user_mlp.add_module('user_relu_%d' % i, nn.ReLU())
        for i in range(len(item_mlp_dims) - 1):
            self.item_mlp.add_module('item_linear_%d' % i, nn.Linear(item_mlp_dims[i], item_mlp_dims[i + 1]))
            self.item_mlp.add_module('item_relu_%d' % i, nn.ReLU())

        self.pred_layer = nn.Sequential(
            nn.Linear(user_mlp_dims[-1], 1),
            nn.Sigmoid(),
        )

        self.device = device
        self.to(device)

    def forward(self, user_vecs, item_vecs):
        user_latent = self.user_mlp(user_vecs)
        item_latent = self.item_mlp(item_vecs)
        dmf_vecs = user_latent * item_latent

        pred_ratings = self.pred_layer(dmf_vecs)
        return pred_ratings

    def predict(self, user_vecs, item_vecs):
        with torch.no_grad():
            pred_ratings = self.forward(user_vecs, item_vecs)

        return pred_ratings
