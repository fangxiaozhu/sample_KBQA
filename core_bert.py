import sys
import codecs
import time
import json
from scipy.spatial.distance import cosine
import code
from models import BertMatch 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pkuseg
from elmoformanylangs import Embedder
import numpy as np
from pytorch_transformers import BertTokenizer
import six
import pandas as pd
from config import config

seg = pkuseg.pkuseg()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertMatch()
model.load_state_dict(torch.load(config.bert_train_modelPath))
model = model.to(DEVICE)
pretrained_model_name_or_path = config.bert_pretrain_model_path
print('bert预训练模型已加载')
class answerCandidate:
    def __init__(self, sub = '', pre = '', qRaw = '', qType = 0, score = 0, kbDict = [], wS = 1, wP = 10, wAP = 100):
        self.sub = sub # subject 
        self.pre = pre # predicate
        self.qRaw = qRaw # raw question
        self.qType = qType # question type
        self.score = score # 分数
        self.kbDict = kbDict # kd dictionary
        self.origin = '' 
        self.scoreDetail = [0,0,0,0,0]
        self.wS = wS # subject的权重
        self.wP = wP # oredicate的权重
        self.wAP = wAP # answer pattern的权重
        self.scoreSub = 0
        self.scoreAP = 0
        self.scorePre = 0
        
    def calcScore(self, qtList, countCharDict, debug=False, includingObj = [], use_elmo=False):
        # 最重要的部分，计算该答案的分数
        lenSub = len(self.sub)
        scorePre = 0
        scoreAP = 0
        pre = self.pre
        q = self.qRaw
        subIndex = q.index(self.sub)
        qWithoutSub1 = q[:subIndex] # subject左边的部分
        qWithoutSub2 = q[subIndex+lenSub:] # subject右边的部分

        qWithoutSub = q.replace(self.sub,'') # 去掉subject剩下的部分
        qtKey = (self.qRaw.replace(self.sub,'(SUB)',1) + ' ||| ' + pre) # 把subject换成(sub)然后加上predicate
        if qtKey in qtList:
            scoreAP = qtList[qtKey] # 查看当前的问题有没有在知识库中出现过
        self.scoreAP = scoreAP
        qWithoutSubSet1 = set(qWithoutSub1)
        qWithoutSubSet2 = set(qWithoutSub2)
        qWithoutSubSet = set(qWithoutSub)
        preLowerSet = set(pre.lower())
        
        # 找出predicate和问题前后两部分的最大intersection
        intersection1 = qWithoutSubSet1 & preLowerSet
        intersection2 = qWithoutSubSet2 & preLowerSet
        if len(intersection1) > len(intersection2):
            maxIntersection = intersection1
        else:
            maxIntersection = intersection2

        # 计算来自predicate的分数，采用最大overlap的character的倒数 1/(n+1)
        preFactor = 0
        for char in maxIntersection:
            if char in countCharDict:
                preFactor += 1/(countCharDict[char] + 1)
            else:
                preFactor += 1

        if len(pre) != 0:
            scorePre = preFactor / len(qWithoutSubSet | preLowerSet)
        else:
            scorePre = 0

        
        if len(includingObj) != 0 and scorePre == 0:
            for objStr in includingObj:
                scorePreTmp = 0
                preLowerSet = set(objStr.lower())
                intersection1 = qWithoutSubSet1 & preLowerSet
                intersection2 = qWithoutSubSet2 & preLowerSet

                if len(intersection1) > len(intersection2):
                    maxIntersection = intersection1
                else:
                    maxIntersection = intersection2

                preFactor = 0
                for char in maxIntersection:
                    if char in countCharDict:
                        preFactor += 1/(countCharDict[char] + 1)
                    else:
                        preFactor += 1

                scorePreTmp = preFactor / len(qWithoutSubSet | preLowerSet)
                if scorePreTmp > scorePre:
                    scorePre = scorePreTmp

        if use_elmo and len(pre) != 0:
            preCut = [pre]
            qWithoutSubCut = [qWithoutSub]
            data_df = pd.DataFrame({'question': qWithoutSubCut, 'sim_question': preCut})
            q_data = data_df['question'].apply(sent2ids)
            p_data = data_df['sim_question'].apply(sent2ids)
            q_data_e = data2tensor([q_data[0]])
            p_data_e = data2tensor([p_data[0]])
            output = model(q_data_e,p_data_e) # 
            scorePre_m = output[0][0]##bert模型做相似度计算，第一个值是相似性分数
            self.scorePre = scorePre_m        

        scoreSub = 0 

        # 计算subject的权重有多高，可能有些subject本身就是更重要一些，一般来说越罕见的entity重要性越高
        for char in self.sub:
            if char in countCharDict:
                scoreSub += 1/(countCharDict[char] + 1)
            else:
                scoreSub += 1

        self.scoreSub = scoreSub
        self.scorePre = scorePre

        self.score = scoreSub * self.wS + scorePre * self.wP + scoreAP * self.wAP
        
        return self.score


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

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
def sent2ids(sent_text):
    sent_tokens = ['[CLS]'] + tokenizer.tokenize(sent_text) + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(sent_tokens)
    return token_ids

def getAnswer(sub, pre, kbDict):
    answerList = []
    for kb in kbDict[sub]:
        if pre in kb:
            answerList.append(kb[pre])
   
    return answerList

def answerQ (qRaw, lKey, kbDict, qtList, countCharDict, wP=10, threshold=0, debug=False):
    q = qRaw.strip().lower() # 问题转化成小写
    candidateSet = set()
    result = ''
    maxScore = 0
    bestAnswer = set()
    for key in lKey:
        if -1 != q.find(key): # 如果问题中出现了该subject，那么我们就要考虑这个subject的triples
            for kb in kbDict[key]:
                for pre in list(kb):
                    newAnswerCandidate = answerCandidate(key, pre, q, wP=wP) # 构建一个新的answer candidate
                    candidateSet.add(newAnswerCandidate)
   
    candidateSetCopy = candidateSet.copy()
    if debug:
        print('len(candidateSet) = ' + str(len(candidateSetCopy)), end = '\r', flush=True)
    candidateSet = set()
    candidateSetIndex = set()

    for aCandidate in candidateSetCopy:
        strTmp = str(aCandidate.sub+'|'+aCandidate.pre)
        if strTmp not in candidateSetIndex:
            candidateSetIndex.add(strTmp)
            candidateSet.add(aCandidate)

    # 针对每一个candidate answer，计算该candidate的分数，然后选择分数最高的作为答案
    for aCandidate in candidateSet:
        scoreTmp = aCandidate.calcScore(qtList, countCharDict,debug)
        if scoreTmp > maxScore:
            maxScore = scoreTmp
            bestAnswer = set()
        if scoreTmp == maxScore:
            bestAnswer.add(aCandidate)
    
    # 去除一些重复的答案        
    bestAnswerCopy = bestAnswer.copy()
    bestAnswer = set()
    for aCandidate in bestAnswerCopy:
        aCfound = 0
        for aC in bestAnswer:
            if aC.pre == aCandidate.pre and aC.sub == aCandidate.sub:
                aCfound = 1
                break
        if aCfound == 0:
            bestAnswer.add(aCandidate)

    # 加入object的分数
    bestAnswerCopy = bestAnswer.copy()
    for aCandidate in bestAnswerCopy:
        if aCandidate.score == aCandidate.scoreSub:
            scoreReCal = aCandidate.calcScore(qtList, countCharDict,debug, includingObj=getAnswer(aCandidate.sub, aCandidate.pre, kbDict))
            if scoreReCal > maxScore:
                bestAnswer = set()
                maxScore = scoreReCal
            if scoreReCal == maxScore:
                bestAnswer.add(aCandidate)

    # 加入cosine similarity
    bestAnswerCopy = bestAnswer.copy()
    if len(bestAnswer) > 1: # use word vector to remove duplicated answer
        for aCandidate in bestAnswerCopy:
            scoreReCal = aCandidate.calcScore(qtList, countCharDict,debug, includingObj=getAnswer(aCandidate.sub, aCandidate.pre, kbDict), use_elmo=True)
            if scoreReCal > maxScore:
                bestAnswer = set()
                maxScore = scoreReCal
            if scoreReCal == maxScore:
                bestAnswer.add(aCandidate)
            
    if debug:
        for ai in bestAnswer:
            for kb in kbDict[ai.sub]:
                if ai.pre in kb:
                    print(ai.sub + ' ' +ai.pre + ' '+ kb[ai.pre])
        return[bestAnswer,candidateSet]       
    else:
        return bestAnswer
        


def loadQtList(path, encode = 'utf8'):
    qtList = json.load(open(path,'r',encoding=encode))
    return qtList

def loadcountCharDict(path, encode = 'utf8'):
    countCharDict = json.load(open(path,'r',encoding=encode))
    return countCharDict

def answerAllQ(pathInput, pathOutput, lKey, kbDict, qtList, countCharDict, qIDstart=1, wP=10):
    fq = open(pathInput, 'r', encoding='utf8')
    i = qIDstart
    timeStart = time.time()
    fo = open(pathOutput, 'w', encoding='utf8')
    fo.close()
    listQ = []
    for line in fq:
        if line[1] == 'q':
            listQ.append(line[line.index('\t')+1:].strip())
    for q in listQ:
        fo = open(pathOutput, 'a', encoding='utf8')
        result = answerQ(q, lKey, kbDict, qtList, countCharDict, wP=wP)
        fo.write('<question id='+str(i)+'>\t' + q.lower() + '\n')
        answerLast = ''
        if len(result) != 0:
            answerSet = []
            fo.write('<triple id='+str(i)+'>\t')
            for res in result:
                answerTmp = getAnswer(res.sub, res.pre, kbDict)
                answerSet.append(answerTmp)
                fo.write(res.sub.lower() + ' ||| ' + res.pre.lower() + ' ||| '\
                         + str(answerTmp)  + ' ||| ' + str(res.score) + ' ====== ')
            fo.write('\n')
            fo.write('<answer id='+str(i)+'>\t')

            answerLast = answerSet[0][0]
            mulAnswer = False
            for ansTmp in answerSet:
                for ans in ansTmp:
                    if ans != answerLast:
                        mulAnswer = True
                        continue
                if mulAnswer == True:
                    continue

            if mulAnswer == True:
                for ansTmp in answerSet:
                    for ans in ansTmp:
                        fo.write(ans)
                        if len(ansTmp) > 1:
                            fo.write(' | ')
                    if len(answerSet) > 1:
                        fo.write(' ||| ')
            else:
                fo.write(answerLast)
                
            fo.write('\n==================================================\n')
        else:
            fo.write('<triple id='+str(i)+'>\t')
            fo.write('\n')
            fo.write('<answer id='+str(i)+'>\t')
            fo.write('\n==================================================\n')
        print('processing ' + str(i) + 'th Q.\tAv time cost: ' + str((time.time()-timeStart) / i)[:6] + ' sec', end = '\r', flush=True)
        fo.close()
        i += 1
    fq.close()       
    

def loadResAndanswerAllQ(pathInput, pathOutput, pathDict, pathQt, pathCD, encode='utf8', qIDstart=1, wP=10):
    kbDict = json.load(open(pathDict, 'r', encoding=encode)) # kbJson.cleanPre.alias.utf8
    qtList = loadQtList(pathQt, encode) # outputAP
    countCharDict = loadcountCharDict(pathCD) # countChar
    answerAllQ(pathInput, pathOutput, list(kbDict), kbDict, qtList, countCharDict, qIDstart=1,wP=wP)


if __name__ == '__main__':
    pathInput = config.test_data_path
    pathOutput = config.result_bert_path
    pathDict = config.kb_process_path
    pathQt = config.output_data_path
    pathCD = config.countChar_dir
    qIDstart = 1
    defaultWeightPre = 30
    loadResAndanswerAllQ(pathInput,pathOutput,pathDict,pathQt,pathCD,'utf8', qIDstart, defaultWeightPre)
