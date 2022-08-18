from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Tuple, Dict, List, Union
from utils.training import configure_optimizers_alon
import torch.nn as nn
import torch.nn.functional as F
from utils.matrics import Statistic
import pprint as pp



class MLP_DATASET(Dataset):
    def __init__(self, data):
        self.x = data['x']
        self.y = data['y']
        self.id = data['id']
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)
    @staticmethod
    def get_data(app_vector, app_label):
        train_data = {'x':[], 'y':[], 'id':[]}
        for k,v in app_vector.items():
            train_data['x'].append(list(v))
            train_data['y'].append(app_label[k])
            train_data['id'].append(k)

        return train_data
    def __getitem__(self, index):

        return self.x[index], self.y[index], self.id[index]

 
    def __len__(self):
        
        return len(self.x)
  
        
class MLP(nn.Module):
    def __init__(
        self,
        config,
        ds_idx,
        input
    ):
        self.ds_idx = ds_idx
        self.input = input
        super().__init__()
        self._GPU = 0
        self._config = config
        self.device =  torch.device("cuda:{}".format(self._GPU) if torch.cuda.is_available() else "cpu")
        self.linear = nn.Sequential(
            nn.Linear(self.input, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2),
        )
        self.loss = F.nll_loss
        if torch.cuda.is_available():
            self.cuda(self._GPU)
        self.to(self.device)
        
        optimizers, schedulers = configure_optimizers_alon(self._config.hyper_parameters,
                                         self.parameters())
        self.optimizer, self.scheduler = optimizers[0], schedulers[0]
        self.result = dict()
        self.loss_dict = dict()

    def forward(self, batch):
        x = self.linear(batch)
        logits = torch.log_softmax(x, dim=-1)
        return logits
    
    
    def get_dataloader(self, dataset, shuffle=False):
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=32,
            num_workers=1,
            shuffle=shuffle
        )
        return dataloader
    
    def fit(self, train_dataset):
        train_dataloader = self.get_dataloader(train_dataset, True)
        self.train()
        for epoch in tqdm(range(10)):
            outs = []
            for x, Y, _id in train_dataloader:
                x = x.to(self.device)
                Y = Y.to(self.device)
                # id = id.to(self.device)
                self.optimizer.zero_grad()
                output = self(x)
                loss = F.nll_loss(output, Y, weight=None)
                # Record loss for each app
                for logit_t, y_t, xfg_id in zip(output, Y, _id):
                    logit = logit_t.tolist()
                    y = y_t.tolist()
                    # xfg_id = xfg_id_t.tolist()
                    if y == 0:
                        loss_v = pow(logit[y] - y, 2)
                    else:
                        loss_v = pow(logit[y] - (y - 1), 2)

                    if xfg_id in self.ds_idx:
                        if str(xfg_id) not in self.loss_dict.keys():
                            self.loss_dict[str(xfg_id)] = list()
                        self.loss_dict[str(xfg_id)].append(loss_v)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step() 
                with torch.no_grad():
                    _, preds = output.max(dim=1)
                    statistic = Statistic().calculate_statistic(
                        Y,
                        preds,
                        2,
                    )
                outs.append({"loss": loss, "statistic": statistic})
            self._general_epoch_end(outs, 'train')
    
    def test(self, test_dataset):
        test_dataloader = self.get_dataloader(test_dataset, False)
        self.eval()
        outs = []
        outputs = []
        for x, y in tqdm(test_dataloader):
           
            x = x.to(self.device)
            y = y.to(self.device)
            output = self(x)
            outputs.append(output)
            loss = F.nll_loss(output, y, weight=None)
            with torch.no_grad():
                _, preds = output.max(dim=1)
                statistic = Statistic().calculate_statistic(
                    y,
                    preds,
                    2,
                )
            outs.append({"loss": loss, "statistic": statistic})
        outputs = torch.cat(outputs, dim=0)
        # Convert to probabilities and return the numpy array of shape N x K
        out = outputs.cpu().detach().numpy() if self.cuda else outputs.detach().numpy()
        pred = np.exp(out)
        
        self._general_epoch_end(outs, 'test')
        return pred
        
    def _general_epoch_end(self, outputs: List[Dict], group: str) -> Dict:
        with torch.no_grad():
            mean_loss = torch.stack([out["loss"]
                                     for out in outputs]).mean().item()
            logs = {f"{group}/loss": mean_loss}
            logs.update(
                Statistic.union_statistics([
                    out["statistic"] for out in outputs
                ]).calculate_metrics(group))
            pp.pprint(logs)
            
