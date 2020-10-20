import argparse
from collections import Counter
import code
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import SNLIModel

import numpy as np
import pandas as pd
import pkuseg
from config import config
from elmoformanylangs import Embedder

e = Embedder(config.elmo_pretrain_model_path)

seg = pkuseg.pkuseg()

allPre = []
with open(config.kb_dir_path) as fin:
    for line in fin:
        allPre.append(line.split(" ||| ")[1])

def getAnswerPatten(inputPath = config.train_data_path):
    inputEncoding = 'utf8'

    data = []
    with open(inputPath, 'r', encoding=inputEncoding) as fi:
        pattern = re.compile(r'[·•\-\s]|(\[[0-9]*\])') #pattern to clean predicate, in order to be consistent with KB clean method
        for line in fi:
            if line.find('<q') == 0:  #question line
                qRaw = line[line.index('>') + 2:].strip()
                continue
            elif line.find('<t') == 0:  #triple line
                triple = line[line.index('>') + 2:]
                s = triple[:triple.index(' |||')].strip()
                triNS = triple[triple.index(' |||') + 5:]
                p = triNS[:triNS.index(' |||')]
                p, num = pattern.subn('', p)
                if qRaw.find(s) != -1:
                    qRaw = qRaw.replace(s,'', 1)
               
                data.append([qRaw, p])
         
            else: continue
    return data

data = getAnswerPatten()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SNLIModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())


num_epochs = 30
batch_size = 64
margin = 1.0
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    print("starting epoch {}".format(epoch))
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]

        questions = [q for q, _ in batch_data]
        predicates = [p for _, p in batch_data]
        negative_predicates = []
        for p in predicates:
            neg_pre = random.sample(allPre, 1)[0]
            while neg_pre == p:
                neg_pre = random.sample(allPre, 1)[0]
            negative_predicates.append(neg_pre)

        questions = [seg.cut(sent) for sent in questions]
        q_embeddings = np.concatenate([np.expand_dims(emb.mean(1), 0) for emb in e.sents2elmo(questions, -2)]) # 3 * 1024

        predicates = [seg.cut(sent) for sent in predicates]
        p_embeddings =  np.concatenate([np.expand_dims(emb.mean(1), 0) for emb in e.sents2elmo(predicates, -2)])
        
        negative_predicates = [seg.cut(sent) for sent in negative_predicates]
        neg_p_embeddings =  np.concatenate([np.expand_dims(emb.mean(1), 0) for emb in e.sents2elmo(negative_predicates, -2)])

        q_embeddings = torch.Tensor(q_embeddings).to(DEVICE)
        p_embeddings = torch.Tensor(p_embeddings).to(DEVICE)
        neg_p_embeddings = torch.Tensor(neg_p_embeddings).to(DEVICE)
       
        pos_score = model(q_embeddings, p_embeddings)#
        neg_score = model(q_embeddings, neg_p_embeddings)
       
        scores = torch.cat([pos_score, neg_score], 1)
        labels = torch.zeros(scores.shape[0]).to(scores.device).long()

        loss = loss_fn(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i // batch_size % 100 ==0:
            print("batch {} loss {}".format(i // batch_size, loss.item()))

    model_path = "snli_checkpoints/elmo_sim_model_epoch{}.pt".format(epoch)
    print("saving model to {}".format(model_path))
    torch.save(model.state_dict(), model_path)
