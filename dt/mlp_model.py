from typing import Tuple, Dict, List, Union
from sklearn.base import BaseEstimator
from tqdm import tqdm
import torch
from omegaconf import DictConfig
from utils.training import configure_optimizers_alon
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Tuple, Dict, List, Union
from omegaconf import DictConfig
from utils.training import configure_optimizers_alon
import torch.nn as nn
import torch.nn.functional as F
from utils.matrics import Statistic
from utils.json_ops import read_json
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
        # liner_input = read_json(path=path)
        train_data = {'x':[], 'y':[], 'id':[]}
        # test_data = {'x':[], 'y':[], 'id':[]}
        # i = 0
        for k,v in app_vector.items():
            # i += 1
            # if i%5 == 0:
            #     test_data['x'].append(v)
            #     test_data['y'].append(app_label[k])
            #     test_data['id'].append(k)
            # else:
            train_data['x'].append(list(v))
            train_data['y'].append(app_label[k])
            train_data['id'].append(k)
        # train_data['x'] = liner_input['train']['liner_input_train']
        # train_data['y'] = liner_input['train']['s']
        # test_data['x'] = liner_input['test']['liner_input_test']
        # test_data['y'] = liner_input['test']['s']
        return train_data
    def __getitem__(self, index):
        
        # if isinstance(index, int):
        return self.x[index], self.y[index], self.id[index]
        
        # elif isinstance(index, list):
        #     x=
        #     for i in index:
        #         self.__getitem__(i)
 
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
            # nn.Linear(1000, 500),
            # nn.ReLU(),
            nn.Linear(1000, 2),
            # nn.ReLU(),
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.Linear(32, 2)
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
                # 记录每个app的loss
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
            
            
class CL_MLP(BaseEstimator):
    def __init__(
            self,
            dataset,
            config: DictConfig,
            no_cuda,
            loader = None
            
    ):
        self.config = config
        self.dataset = dataset
        self.batch_size = config.cl.batch_size
        self.epochs = config.cl.n_epochs
        self.lr = config.hyper_parameters.learning_rate
        self.no_cuda = no_cuda
        self.seed = config.seed
        self._GPU = config.gpu
        self.cuda = not self.no_cuda and torch.cuda.is_available()
        self.log_interval = config.hyper_parameters.log_interval
        torch.manual_seed(self.seed)
        if self.cuda:  # pragma: no cover
            torch.cuda.manual_seed(self.seed)

        # Instantiate PyTorch model
        self.model = MLP(self.config)
        #设置GPU编号
        if self.cuda:  # pragma: no cover
            self.model.cuda(self._GPU)

        self.loader_kwargs = {'num_workers': self.config.num_workers,
                              'pin_memory': True} if self.cuda else {}
        self.loader = loader
        
        self.test_batch_size = config.hyper_parameters.test_batch_size
        self.device = torch.device("cuda:{}".format(self._GPU) if torch.cuda.is_available() else "cpu")
        
        
    def get_sub_dataset(self, idx):
        
        sub_dataset = Subset(self.dataset, idx)
        return sub_dataset
     
     
    def fit(self, train_idx, train_labels=None, sample_weight=None,
            loader='train'):
        """This function adheres to sklearn's "fit(X, y)" format for
        compatibility with scikit-learn. ** All inputs should be numpy
        arrays, not pyTorch Tensors train_idx is not X, but instead a list of
        indices for X (and y if train_labels is None). This function is a
        member of the cnn class which will handle creation of X, y from the
        train_idx via the train_loader. """
        if self.loader is not None:
            loader = self.loader
        if train_labels is not None and len(train_idx) != len(train_labels):
            raise ValueError(
                "Check that train_idx and train_labels are the same length.")

        if sample_weight is not None:  # pragma: no cover
            if len(sample_weight) != len(train_labels):
                raise ValueError("Check that train_labels and sample_weight "
                                 "are the same length.")
            class_weight = sample_weight[
                np.unique(train_labels, return_index=True)[1]]
            class_weight = torch.from_numpy(class_weight).float()
            if self.cuda:
                class_weight = class_weight.cuda()
        else:
            class_weight = None
        train_dataset = self.get_sub_dataset(train_idx)
        self.model.fit(train_dataset=train_dataset)
        
        
        
    def predict(self, idx=None, loader=None):
        """Get predicted labels from trained model."""
        # get the index of the max probability
        probs = self.predict_proba(idx)
        return probs.argmax(axis=1)

    
    def predict_proba(self, idx=None, loader=None):
        if self.loader is not None:
            loader = self.loader
        # if loader is None:
        #     is_test_idx = idx is not None and len(
        #         idx) == self.test_size and np.all(
        #         np.array(idx) == np.arange(self.test_size))
        #     loader = 'test' if is_test_idx else 'train'
        test_dataset = self.get_sub_dataset(idx)

        # sets model.train(False) inactivating dropout and batch-norm layers

        # Run forward pass on model to compute outputs
        pred = self.model.test(test_dataset)
        
        return pred