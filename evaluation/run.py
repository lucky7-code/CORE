"""
Usage:
    run.py  [options]

Options:
    --seed=<int>                        random seed [default: 42]
    --fold=<int>                        fold number [default: 0]
    --cuda=<int>                        use GPU id [default: 0]
    --dataset=<string>                  dataset [default: assist2009]#nips_task34
    --model=<string>                    model type [default: akt]
    --CORE=<int>                        whether use CORE or not  0:False 1:True[default: 1]
    --state=<string>                    model state [default: test]
    --balance=<int>                     whether test dataset balance or not 0:False 1:True[default: 1]
"""
import json
import os
import random
import logging
import torch
import numpy as np
from datetime import datetime
from docopt import docopt
from torch.optim import SGD
from torch.optim.adam import Adam

from data_loading.getdata import getdata
from model.init_model import init_model
from evaluation import eval


def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    global model
    args = docopt(__doc__)
    seed = int(args['--seed'])
    cuda = args['--cuda']
    model_type = args['--model']
    CORE = True if int(args['--CORE']) == 1 else False
    balance = True if int(args['--balance']) == 1 else False
    fold = int(args['--fold'])
    state = args['--state']
    dataset = args['--dataset']
    print(args)
    logger = logging.getLogger('main')
    logger.setLevel(level=logging.DEBUG)
    date = datetime.now()
    handler = logging.FileHandler(
        f'log/{date.year}_{date.month}_{date.day}_{model_type}_result.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(model_type)
    logger.info(list(args.items()))

    setup_seed(seed)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        device = torch.device('cuda',int(cuda))
    else:
        device = torch.device('cpu')
    with open("../configs/kt_config.json") as f:
        config = json.load(f)
        train_config = config["train_config"]
        model_config = config[model_type]
        if model_type in ["dkvmn", "sakt", "saint", "akt", "dkt"]:
            train_config["batch_size"] = 64
    batch_size, num_epochs, optimizer,learning_rate = train_config["batch_size"], train_config["num_epochs"], train_config[
            "optimizer"], model_config["learning_rate"]
    with open("../configs/data_config.json") as fin:
        data_config = json.load(fin)
    emb_type = 'qid'
    del model_config["learning_rate"]
    model = init_model(model_type, CORE, model_config, data_config[dataset], emb_type,device)
    ckpt = 'checkpoint/' + model_type + '/' + ("CORE_" if CORE else "") + model_type + (
        "_balance_" if balance else "_unbalance_") + dataset + '.pth.tar'
    if state=='train':
        trainLoader, validLoader, testLoader = getdata(dataset, data_config, fold, batch_size)
        opt = Adam(model.parameters(), learning_rate)
        print(f"model_config:  {model_config}")
        print(f"train_config:  {train_config}")
        print(f"model is {model}")
        best_auc = 0
        best_acc = 0
        best_epoch = 0
        for epoch in range(1,num_epochs+1):
            print('epoch: ' + str(epoch))
            model = eval.train_epoch(model, trainLoader, opt)
            logger.info(f'epoch {epoch}')
            if balance:
                state = ["Valid",data_config[dataset]["num_q"],balance]
            else:
                state = ["Valid",data_config[dataset]["num_q"],balance]
            auc, acc = eval.test_epoch(model, validLoader,device, state)
            if auc > best_auc+1e-3:
                logger.info('best checkpoint')
                print('best checkpoint')
                torch.save({'state_dict': model.state_dict()}, ckpt)
                test_auc, test_acc = eval.test_epoch(model, testLoader, device, state)
                best_auc = auc
                best_acc = acc
                best_epoch = epoch
                print('best_test_auc ' + str(test_auc) + '   best_test_acc ' + str(test_acc) )
            print('best_auc '+str(best_auc)+'   best_acc '+str(best_acc)+'   best_epoch '+str(best_epoch))

            logger.info('best_auc ' + str(best_auc) + '   best_acc ' + str(best_acc) + '   best_epoch ' + str(best_epoch))

            if epoch - best_epoch >= 10:
                if balance:
                    state = ["Test", data_config[dataset]["num_q"], balance]
                else:
                    state = ["Test", data_config[dataset]["num_q"], balance]
                test_auc, test_acc = eval.test_epoch(model, testLoader,device, state, ckpt=ckpt)
                print('test_auc ' + str(test_auc) + '   test_acc ' + str(test_acc))
                logger.info('test_auc ' + str(test_auc) + '   test_acc ' + str(test_acc))
                break
    else:
        if balance:
            state = ["Test", data_config[dataset]["num_q"], balance]
        else:
            state = ["Test", data_config[dataset]["num_q"], balance]
        testLoader = getdata(dataset, data_config, -1, batch_size, True)
        auc, acc = eval.test_epoch(model, testLoader, device,state, ckpt=ckpt)
        print('test_auc ' + str(auc) + '   acc ' + str(acc))


if __name__ == '__main__':
    main()
