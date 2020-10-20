import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import code
from pytorch_transformers import BertModel
from config import config

class SNLIModel(nn.Module):
    def __init__(self, hidden_size=1024, num_layers=3):
        super(SNLIModel, self).__init__()
        self.layer_weights = nn.Parameter(torch.ones(num_layers))
        self.linear1 = nn.Linear(4*hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x, y):
        layer_weights = torch.exp(F.log_softmax(self.layer_weights)).unsqueeze(0).unsqueeze(2)
        x = (layer_weights * x).sum(1)
        y = (layer_weights * y).sum(1)
        return F.tanh(self.linear2(F.relu(self.linear1(torch.cat([x, y, torch.abs(x-y), x*y], 1)))))


class WeightedAvgModel(nn.Module):
    def __init__(self, hidden_size=1024, num_layers=3):
        super(WeightedAvgModel, self).__init__()
        self.layer_weights = nn.Parameter(torch.ones(num_layers)) 
        
    def forward(self, x):
        layer_weights = torch.exp(F.log_softmax(self.layer_weights)).unsqueeze(0).unsqueeze(2)
        x = (layer_weights * x).sum(1)  
        return x


class BertMatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_pretrain_model_path)
        self.soft_max = nn.Softmax(dim=-1)
        self.fc = nn.Linear(768 * 3, 2)
        self.dropout = nn.Dropout(0.5)
        self.loss = nn.CrossEntropyLoss()

    def encode(self, x):
        outputs = self.bert(x)
        pooled_out = outputs[1]
        return pooled_out

    def forward(self, input1_ids, input2_ids, labels=None):
        seq1_output = self.encode(input1_ids)
        seq2_output = self.encode(input2_ids)
        feature = torch.cat([seq1_output, seq2_output, seq1_output - seq2_output], dim=-1)
        logistic = self.fc(feature)
        if labels is None:
            output = self.soft_max(logistic)
            pred = torch.argmax(output, dim=-1)
            return output
        else:
            loss = self.loss(logistic, labels)
            return logistic, loss