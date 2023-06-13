import tqdm
import torch
import logging
import os
from sklearn import metrics
import numpy as np
logger = logging.getLogger('main.eval')
def __load_model__(ckpt):
    '''
    ckpt: Path of the checkpoint
    return: Checkpoint dict
    '''
    if os.path.isfile(ckpt):
        checkpoint = torch.load(ckpt)
        print("Successfully loaded checkpoint '%s'" % ckpt)
        return checkpoint
    else:
        raise Exception("No checkpoint found at '%s'" % ckpt)


def train_epoch(model, trainLoader, optimizer):
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        optimizer.zero_grad()
        loss, _, _, _ = model(batch)
        loss.sum().backward()
        optimizer.step()
    return model


def test_epoch(model, testLoader, device, state, ckpt=None):
    if ckpt is not None:
        checkpoint = __load_model__(ckpt)
        if 'mem.memory_value' in list(checkpoint['state_dict'].keys()):
            del checkpoint['state_dict']['mem.memory_value']
        model.load_state_dict(checkpoint['state_dict'])
    ground_truth = torch.tensor([], device="cuda")
    prediction = torch.tensor([], device="cuda")
    prediction_1 = torch.tensor([], device=device)
    prediction_2 = torch.tensor([], device=device)
    skill = torch.tensor([], device=device)
    with torch.no_grad():
        for batch in tqdm.tqdm(testLoader, desc=state[0]+'ing:     ', mininterval=2):
            loss, p, label, s = model(batch)
            prediction = torch.cat([prediction, p])
            ground_truth = torch.cat([ground_truth, label])
            skill = torch.cat([skill, s])
    prediction = prediction.detach().cpu().numpy()
    ground_truth = ground_truth.detach().cpu().numpy()
    if state[2]:
        prediction, ground_truth = Balance_Example(prediction, ground_truth, skill.detach().cpu().numpy(),state[1])

    acc = metrics.accuracy_score(np.round(ground_truth), np.round(prediction))
    auc = metrics.roc_auc_score(ground_truth, prediction)
    logger.info('auc: ' + str(auc) + ' acc: ' + str(acc))
    print('auc: ' + str(auc) + ' acc: ' + str(acc))
    return auc, acc


def Balance_Example(ori_pre, ori_groundTruth, ori_skill, example_num):
    '''
    input:
        ori_pre: Original forecast    Dim :[N]
        ori_groundTruth: Original groundTruth   Dim :[N]
        ori_skill: Original skill   Dim :[N]
        example_num: number of need balance example  Int
        each_Example_num: number of need balance each example  Int
    '''
    '''
    return:
        pre:Processed pre      Dim :[k]
        groundTruth: Processed groundTruth    Dim :[k]
    '''
    pre = np.asarray([])
    groundTruth = np.asarray([])
    '''bias 强度设置'''
    # q = np.loadtxt("p09.csv")
    # Q_indexs = np.where((q>=0.8)|(q<=0.2))[0]
    # Q_indexs = np.where(((q>0.6)&(q<0.8))|((q>0.2)&(q<0.4)))[0]
    # Q_indexs = np.where((q>=0.4)&(q<=0.6))[0]
    # for i in Q_indexs:
    
    for i in range(example_num):
        true_Index = np.where(ori_skill==i)[0]
        false_Index = np.where(ori_skill==(i+example_num))[0]

        if len(true_Index)== 0 or len(false_Index)==0:
            continue
        num_true = (len(true_Index) + len(false_Index)) // 2
        num_false = num_true
        balance_True_Index = np.random.choice(true_Index, num_true)
        balance_False_Index = np.random.choice(false_Index, num_false)
        balance_Index = np.concatenate([balance_True_Index, balance_False_Index])

        pre = np.concatenate([pre, ori_pre[balance_Index]])
        groundTruth = np.concatenate([groundTruth, ori_groundTruth[balance_Index]])
    return pre, groundTruth