# -*- coding: UTF-8 -*-
#coding=utf8
import logging
import os
from pathlib import Path
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = cur_dir
input_data_dir = os.path.join(root_dir, "input_data")
result_data_dir = os.path.join(root_dir, 'result_data')
model_data_dir = os.path.join(root_dir, 'model_data')

train_data_path = os.path.join(input_data_dir, "training-data")
test_data_path = os.path.join(input_data_dir, "test-data")
kb_dir_path = os.path.join(input_data_dir, "nlpcc-iccpol-2016.kbqa.kb")
word2vec_dict_path = os.path.join(input_data_dir, "sgns.wiki.bigram-char")

countChar_dir = os.path.join(result_data_dir, "countChar")
countChar_txt_dir = os.path.join(result_data_dir, 'countChar.txt')
kb_process_path = os.path.join(result_data_dir, 'kbJson.cleanPre.alias.utf8')
word2vec_process_path = os.path.join(result_data_dir, 'vectorJson.utf8')
output_data_path = os.path.join(result_data_dir, 'outputAP')
result_path = os.path.join(result_data_dir, 'answer')
result_snli_path = os.path.join(result_data_dir, 'answer_snli')
result_weightedAvg_path = os.path.join(result_data_dir, 'answer_weightedAvg')
result_bert_path = os.path.join(result_data_dir, 'answer_bert')

elmo_pretrain_model_path = os.path.join(model_data_dir, 'ELMoForManyLangs/configs/zhs.model')
weightedAvg_model_path = os.path.join(model_data_dir, 'wv_checkpoints/weightedAvg_model_epoch19.pt')
snli_model_path = os.path.join(model_data_dir, 'snli_checkpoints/elmo_sim_model_epoch19.pt')
bert_pretrain_model_path = os.path.join(model_data_dir, 'bert-base-model-pytorch')
bert_train_modelPath = os.path.join(model_data_dir, 'bert_checkpoints/bert_model_epoch3.pt')