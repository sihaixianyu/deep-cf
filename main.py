import os.path
import time

import toml
import torch.nn
from torch.utils.data import DataLoader

from dataset import TestDataset
from dataset import TrainDataset
from evaluator import Evaluator
from model import MLP
from trainer import Trainer
from util import print_res, load_data

root = './'
# root = 'drive/MyDrive/deep-cf/'

if __name__ == '__main__':
    config = toml.load(os.path.join(root, 'config.toml'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_prefix = os.path.join(root, 'data/', config['data_name'])
    user_num, item_num, inter_mat, train_list, test_list = load_data(data_prefix)

    train_dataset = TrainDataset(user_num, item_num, inter_mat, train_list, config['neg_num'])
    test_dataset = TestDataset(user_num, item_num, inter_mat, test_list)

    # Warning: we can't not shuffle the test data, because it has static oreder for each user
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['test_neg_num'], shuffle=False)

    model = MLP(user_num, item_num, config['mlp_dims'], device)
    # model = DMF(user_num, item_num, config['user_mlp_dims'], config['item_mlp_dims'], device)
    # model = CFNet(user_num, 'item_num', config['mlp_dims'], config['user_mlp_dims'], config['item_mlp_dims'], device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['lambda'])

    trainer = Trainer(train_loader, model, optimizer)
    evaluator = Evaluator(test_loader, model, topk=config['topk'])

    best_epoch = {
        'epoch': 0,
        'hit': .0,
        'ndcg': .0,
    }
    for epoch in range(1, config['epoch_num'] + 1):
        train_start = time.time()
        # loss = trainer.train()
        loss = 0
        train_time = time.time() - train_start

        eval_start = time.time()
        hit, ndcg = evaluator.evaluate()
        eval_time = time.time() - eval_start

        print('Epoch=%3d, Loss=%.4f, Hit=%.4f, NDCG=%.4f, Time=(%.4f + %.4f)'
              % (epoch, loss, hit, ndcg, train_time, eval_time), end='\n\n')

        if best_epoch['hit'] <= hit:
            best_epoch['epoch'] = epoch
            best_epoch['hit'] = hit
            best_epoch['ndcg'] = ndcg

    print_res('Best Epoch=%.4f, Hit=%.4f, NDCG=%.4f,'
              % (best_epoch['epoch'], best_epoch['hit'], best_epoch['ndcg']))
