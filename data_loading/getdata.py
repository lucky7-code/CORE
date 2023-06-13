import json

from data_loading.dataloader import KTDataset
from data_loading.que_data_loader import KTQueDataset
from torch.utils.data import DataLoader
import os
def getdata(dataset_name, data_config, i, batch_size, is_test=False):
    data_config = data_config[dataset_name]
    all_folds = set(data_config["folds"])
    if is_test:
        # curtest = KTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
        curtest = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]), data_config["input_type"],
                            {-1}, data_config["num_c"], data_config["max_concepts"])
        test_loader = DataLoader(curtest, batch_size=batch_size)
        return test_loader
    else:
        # curvalid = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]),
        #                      data_config["input_type"], {i})
        # curtrain = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]),
        #                      data_config["input_type"], all_folds - {i})
        # curtest = KTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
        curvalid = KTQueDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                             data_config["input_type"], {i}, data_config["num_c"], data_config["max_concepts"])
        curtrain = KTQueDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                             data_config["input_type"], all_folds - {i}, data_config["num_c"], data_config["max_concepts"])
        curtest = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
                            data_config["input_type"], {-1}, data_config["num_c"], data_config["max_concepts"])
        test_loader = DataLoader(curtest, batch_size=batch_size)
        train_loader = DataLoader(curtrain, batch_size=batch_size)
        valid_loader = DataLoader(curvalid, batch_size=batch_size)
        return train_loader, valid_loader, test_loader

if __name__ == '__main__':
    with open("../configs/data_config.json") as fin:
        data_config = json.load(fin)
    # a,b = getdata("assist2009",data_config,0,256)
    print()