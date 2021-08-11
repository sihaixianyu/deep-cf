import torch
import torch.nn as nn


class CFNet(nn.Module):
    def __init__(self, user_num, item_num, mlp_dims, user_mlp_dims, item_mlp_dims, device='cpu'):
        super(CFNet, self).__init__()
        assert user_mlp_dims[-1] == item_mlp_dims[-1], 'The DMF outlayer must be the same'

        # In the official implement, the first layer has no activation
        self.embed_user = nn.Linear(item_num, mlp_dims[0] // 2)
        self.embed_item = nn.Linear(user_num, mlp_dims[0] // 2)

        self.mlp = nn.Sequential()
        for i in range(len(mlp_dims) - 1):
            self.mlp.add_module('mlp_linear_%d' % i, nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            self.mlp.add_module('mlp_relu_%d' % i, nn.ReLU())

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
            nn.Linear(mlp_dims[-1] + user_mlp_dims[-1], 1),
            nn.Sigmoid(),
        )

        self.device = device
        self.to(device)

    def forward(self, user_vecs, item_vecs):
        user_embedding = self.embed_user(user_vecs)
        item_embedding = self.embed_item(item_vecs)
        mlp_vecs = torch.cat((user_embedding, item_embedding), dim=1)
        mlp_vecs = self.mlp(mlp_vecs)

        user_latent = self.user_mlp(user_vecs)
        item_latent = self.item_mlp(item_vecs)
        dmf_vecs = user_latent * item_latent

        # Concatenate DMF and MLP parts
        pred_vecs = torch.cat((dmf_vecs, mlp_vecs), dim=1)
        pred_ratings = self.pred_layer(pred_vecs)

        return pred_ratings

    def predict(self, user_vecs, item_vecs):
        with torch.no_grad():
            pred_ratings = self.forward(user_vecs, item_vecs)

        return pred_ratings
