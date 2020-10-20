import argparse
from collections import Counter
import code
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import random
from pytorch_transformers import BertTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import BertMatch

###  --将模型改成bert
import numpy as np
import pandas as pd
import pkuseg
from config import config
from elmoformanylangs import Embedder
import numpy as np
import six

pretrained_model_name_or_path = config.bert_pretrain_model_path

seg = pkuseg.pkuseg()

allPre = []
with open(config.kb_dir_path) as fin:
    for line in fin:
        allPre.append(line.split(" ||| ")[1])

train_path = config.train_data_path
test_path = config.test_data_path
num_epochs = 20
batch_size = 32
learning_rate = 0.00002
train_data = getAnswerPatten(train_path)
test_data = getAnswerPatten(test_path)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertMatch().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def getAnswerPatten(inputPath):
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
    

def process_data(batch_size,p_data):
    negative_predicates = []
    questions_new_list = []
    predicates_new_list = []
    label_list = []
    questions_list = [q for q, _ in p_data]
    predicates_list = [p for _, p in p_data]
    
    for p in predicates_list:
        neg_pre = random.sample(predicates_list, 1)[0]
        while neg_pre == p:
            neg_pre = random.sample(predicates_list, 1)[0]
        negative_predicates.append(neg_pre)

    ques_data_list = []
    sim_data_list = []
    for i in range(0,len(questions_list)):
        q = questions_list[i]
        ques_data_list.append(q)
        pos_data =  questions_list[i]+ predicates_list[i]
        sim_data_list.append(pos_data)
        label_list.append(1)
    for i in range(0,len(questions_list)):
        q = questions_list[i]
        ques_data_list.append(q)
        neg_data = questions_list[i]+ negative_predicates[i]
        sim_data_list.append(neg_data)
        label_list.append(0)

    data_df = pd.DataFrame({'ques': ques_data_list, 'sim': sim_data_list, 'label': label_list})
    x_data = data_df['ques'].apply(sent2ids)
    y_data = data_df['sim'].apply(sent2ids)
    label_data = data_df['label']
    data_size = len(ques_data_list)
    
    order = list(range(data_size))
    np.random.shuffle(order)

    for batch_step in range(data_size // batch_size + 1):
        batch_idxs = order[batch_step * batch_size:(batch_step + 1) * batch_size]
        if len(batch_idxs) != batch_size:  # batch size 不可过大; 不足batch_size的数据丢弃（最后一batch）
            continue

        q_sents = data2tensor([x_data[idx] for idx in batch_idxs])
        a_sents = data2tensor([y_data[idx] for idx in batch_idxs])
        batch_labels = data2tensor([label_data[idx] for idx in batch_idxs], pad=False)
        yield q_sents, a_sents, batch_labels


tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
def sent2ids(sent_text):
    sent_tokens = ['[CLS]'] + tokenizer.tokenize(sent_text) + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(sent_tokens)
    return token_ids


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    num_samples = len(sequences)
    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x



def data2tensor(batch_token_ids, pad=True, maxlen=50):
    if pad:
        batch_token_ids = pad_sequences(batch_token_ids, maxlen=maxlen, padding='post')
    batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long).to(DEVICE)
    return batch_token_ids



def evaluate(model):
    model.eval()
    loss = 0.
    total_count = 0.
    with torch.no_grad():
        for q_sents, a_sents, batch_labels in process_data(batch_size,test_data):
            
            with torch.no_grad():
                logistic,loss = model(q_sents, a_sents, batch_labels)
            loss = loss
    model.train()
    return loss


for epoch in range(num_epochs):
    i = 0
    for q_sents, a_sents, batch_labels in process_data(batch_size,train_data):
        logistic,loss = model(q_sents, a_sents, batch_labels)

        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i+=1
        if i% 10 == 0:
            print("epoch", epoch, "iter", i, "loss", loss.item())
        if i% 100 == 0:
	        evaluate_loss = evaluate(model)
	        print('验证集的损失----',evaluate_loss.item())
    model_path = "bert_checkpoints/bert_model_epoch{}.pt".format(epoch)
    print("saving model to {}".format(model_path))
    torch.save(model.state_dict(), model_path)

    